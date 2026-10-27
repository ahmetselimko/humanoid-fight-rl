import warnings
import math
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import amp
import mujoco
import mujoco.viewer
from gymnasium import spaces
import json
import os
import time
import random
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import multiprocessing as mp
from functools import partial
# ThreadPoolExecutor kaldırıldı — GIL nedeniyle CPU-bound MuJoCo'yu gerçek paralel çalıştırmıyor
# CUDA ayarları
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()
NUM_PARALLEL_ENVS = 16

# Ödül ölçeği: PPO critic'i yüzlerce birimlik return'lerle iyi öğrenemez.
# Tüm ödüller toplanırken bu katsayıyla küçültülür (yapı aynı kalır).
REWARD_SCALE = 0.1

# Kafa yüksekliği eşikleri (XML'e göre: ayakta kafa merkezi ~0.79 m,
# dizler doğal esnemeyle ~0.64'e iner — diz eklem aralığı 0'ı içermiyor)
HEAD_UPRIGHT_Z = 0.60   # bunun üstü "dik duruyor"
HEAD_FLOOR_Z   = 0.35   # bunun altı "yerde"
# Gövde dikliği: xmat[8] = gövdenin yerel yukarı ekseninin dünya çerçevesinde
# yukarıya hizalanma derecesi (1.0=tam dik, 0=yatay). ep3300 testinde görüldü:
# öne doğru düşen ajan, düşüşün son anına kadar kafa z>0.60 kalabiliyor
# (momentum + bacak henüz bükülmemiş) — sadece yükseklikle "aktif dik"
# sayılmak bu "önce topla sonra düş" kaçamağını kapatamadı. Gövde açısı,
# öne eğilmeyi yükseklik düşmeden çok önce yakalar.
TORSO_UPRIGHT_MIN = 0.70

# ─────────────────────────────────────────────────────────────────────────
# v2c YÜRÜME-ODAKLI ÖDÜL (literatüre dayalı: MuJoCo Humanoid-v4 tarzı)
# ─────────────────────────────────────────────────────────────────────────
# v2b teşhisi (ep3300/3600/3900, 3 doğrulama döngüsü + rasyonel-intihar +
# lean-trace diagnostikleri): ajanlar YÜRÜYEMİYOR. İlk ~600 adım dik durup
# neredeyse hiç yaklaşmıyor, sonra ani bir öne ATILIM (ileri hız 0.1→1.27 m/s)
# yapıp deviriliyorlar. Kök neden: eski ödül dik durmayı zayıf (+0.3) ödüllendirip
# yaklaşmayı baskın (-0.20×prox_gap mesafe cezası, ~-0.5/adım) hale getirdi →
# politika dengeyi feda edip atıldı. Kanıtlanmış lokomotion RL bunun TERSİNİ
# yapar: dik kalma BASKIN ödül, ileri hareket SINIRLI ikincil ödül.
UPRIGHT_HEALTHY_BONUS = 1.0    # dik+dengeli her adım (BASKIN sinyal). z>UPRIGHT_Z
                              # ve gövde dik iken verilir. Standing≈+1.0/adım.
FORWARD_TARGET_SPEED  = 0.5    # m/s — bu hıza kadar ileri hareket ödüllendirilir;
                              # ÜSTÜ ödüllenmez (doyum). Atılım avantajını yok eder.
FORWARD_REWARD_WEIGHT = 2.5    # ileri ödül = W × clip(ileri_hız, -TARGET, +TARGET).
                              # İleri maks +1.25, GERİYE maks -1.25 (CEZA).
                              # KÖK-NEDEN DÜZELTMESİ (2026-07-16): eski W=1.0 +
                              # geriye-ceza-yok kurulumunda "ayakta dur+1.0 vs
                              # ileri-maks+0.5" dengesizliği vardı → risk-ayarlı
                              # olarak "geriye yaslan ama ayakta kal" (dik %70,
                              # -0.43m) yerel-optimumu politikayı çekti (taze ikiz-
                              # critic bunu açığa çıkardı). Yeni: ileri+1.25 →
                              # toplam +2.25 vs ayakta +1.0 (net baskın) + geriye
                              # AKTİF ceza. (Not: eski "W=2 tek başına" denemesi
                              # YERİNDE-adımlamayı çözmemişti — o farklı bir hata;
                              # buradaki yeni kritik bileşen GERİYE cezası, negatif
                              # hızı hedefler.)
GAIT_ALTERNATION_BONUS = 0.15  # tek-ayak destek DEĞİŞİMİ başına (ep1200 orijinali).
GAIT_CONTACT_Z = 0.12          # ayak bu z altındaysa "yere basıyor" sayılır.

# ─────────────────────────────────────────────────────────────────────────
# v2c/ep1000 SAF LOCOMOTION FAZI (kullanıcı gözlemi + isteği)
# ─────────────────────────────────────────────────────────────────────────
# Gözlem: ajanlar adım atıp yaklaşıyor AMA yaklaşınca yapacak iş olmadığı için
# (dövüş yok) ve dik durmak zaten +1.0 verdiği için "olduğun yerde dur" bir
# YEREL OPTİMUM oluyor — sürekli yürüme gelişemiyor. Çözüm: dövüşü geçici
# duraklat, uzun/sürekli bir yürüme hedefi ver (gerçek Humanoid-v4 tarzı:
# "sürekli ileri yürü + ayakta kal + düşersen KALK"). Hedef varış değil,
# hareketin kendisi. Ajanlar YAN YANA, AYNI yöne (+x) bakacak şekilde başlar
# (çarpışma yok), her biri uzaktaki +x hedefine yürür. Düşünce episode BİTMEZ
# (kalkma öğrensinler). Yürüme oturunca dövüş geri eklenecek.
LOCOMOTION_PHASE = True         # False → normal dövüş moduna döner
# ── DÖVÜŞ FAZI 1: SABİT RAKİBE KARŞI (2026-07-16) ──────────────────────────
# Teşhis: dövüşü DOĞRUDAN self-play ile öğretmek, yürüyücüyü BOZDU — self-play
# non-stationarity (hareketli/değişen rakip) locomotion'un kasıtlı KALDIRDIĞI
# regresyon kök nedeni. ep3200 yürüyücü ep400'de dik %75→%36'ya çöktü, vuruş
# öğrenilmedi. ÇÖZÜM (locomotion reçetesi): önce SABİT/DEĞİŞMEYEN rakibe karşı
# öğren. COMBAT_VS_FROZEN=True iken (dövüş modunda) P2 HER episode DONMUŞ
# rakip (snapshot GÜNCELLENMEZ, ep3200'e kilitli) → non-stationarity YOK.
# Öğrenen ajan (P1) kararsızlık olmadan VURMAYI keşfeder. Vuruş oturunca bu
# bayrak kapatılıp self-play kademeli açılır.
COMBAT_VS_FROZEN = False
# ── DÖVÜŞ FAZI-2: YAKIN-MESAFE MÜFREDATI (2026-07-16) ──────────────────────
# Teşhis: ajanlar rakibe ~1.1-1.3m yaklaşıp DURUYOR, yumruk sallıyor ama TEMAS
# ETMİYOR (~600 dövüş episode, 0 isabet — seyrek-temas keşif problemi). Güçlü
# ceza geri tepti (dik %100→%41). ÇÖZÜM: rakibi ÇOK yakına (~1m, temas menzili)
# spawn et → yumruk HEMEN değer → isabetin DEV ödülü (dmg×2) keşfedilir →
# vuruş pekişir. Vuruş oturunca mesafe kademeli açılır. CLOSE_SPAWN_DIST set
# ise dövüş spawn mesafesini EZER (curriculum difficulty'sinden bağımsız).
CLOSE_SPAWN_DIST = None
# ── AYAĞA-KALKMA (GETUP) FAZI (2026-07-16, kullanıcı isteği) ───────────────
# Sorun: dövüşte/yürümede ajan düşünce episode bitiyor → ayağa KALKMAYI hiç
# öğrenmiyor. GETUP_PHASE=True iken: ajan DÜŞMÜŞ (yerde, yatık) başlar, tek
# görev DİK POZİSYONA gelmek. Yoğun ödül (baş yüksekliği + gövde dikliği) →
# yerdeyken bile yukarı-ilerleme ödüllenir. Fall-termination YOK (tüm episode
# kalkmayı dener). Tek ajan (P2 sıfır aksiyon). Yürüme reçetesinin varyantı —
# net hedefli, seyrek-temas duvarı yok, büyük ihtimalle oturur.
GETUP_PHASE = False
GETUP_MAX_STEPS = 400           # getup: kalkmaya çalışmak için episode süresi
# AYAKTA-BAŞLA + İTME (kullanıcı fikri 2026-08): yapay yatık-spawn yürüyücünün
# zıplama-önyargısını tetikliyordu (ep0 vz=6.4 fırlama). Onun yerine DİK başla,
# rastgele bir İTME uygula → küçük itme toparlanır (kolay +sinyal), büyük itme
# DOĞAL düşürür (getup pratiği, fizik-doğal poz — prone-spawn değil). Episode
# düşünce BİTMEZ → doğal düştüğü pozdan kalkmayı öğrenir. İtme büyüklüğü = müfredat.
GETUP_SPAWN_STANDING = True     # False → eski yatık-spawn (müfredat), True → dik+itme
GETUP_PUSH_MAX = 3.5            # maks itme (m/s + rad/s); uniform(0, bu) → çeşitli şiddet
COMBINE_MAX_STEPS = 800         # birleşik: dövüş → düş → KALK → devam döngüsüne
                                # yeterli süre (400 kısaydı, geç düşünce kalkma
                                # penceresi kalmıyordu — kullanıcı tespiti)
# ── BİRLEŞİK (COMBINE) FAZ: DÖVÜŞ + DOĞAL-DÜŞÜŞTEN KALKMA (2026-07-16, kullanıcı
# fikri) ─────────────────────────────────────────────────────────────────
# İzole getup ÇALIŞMADI (yapay 90° yatık pozisyon → %25 fırlama, %0 kalkma).
# Kullanıcı fikri: dövüşte düşünce episode BİTMESİN, ajan DOĞAL düştüğü
# (ayaktan devrilme = hafif) pozisyonlardan kalkmayı dövüşle BİRLİKTE öğrensin.
# Doğal düşüşler yapay-yatıktan çok daha kolay + kendi içinde "hafif→zor"
# müfredatı barındırır. COMBINE_PHASE=True iken: (1) düşme episode'u BİTİRMEZ,
# (2) düşünce GETUP-GRADYANI (yerden yükselmeyi ödüllendir, anti-fırlama)
# devreye girer, ayaktayken normal dövüş ödülü. Tek politika: ayakta dövüş,
# düşünce kalk. Taban: sparring-dövüşçü (zaten dövüşü kısmen biliyor).
COMBINE_PHASE = False
# ── TAKLİT (IMITATION) — referans kliplerle kalkma-şeklini öğret (DeepMimic-lite).
# Model kalkamıyor; el-gradyanı sadece "yerde kötü/dik iyi" diyor, kalkmanın
# ŞEKLİNİ (dizini topla→çömel→dik) öğretmiyor. USE_IMITATION=True iken COMBINE
# düşme dalına, kalkma-referansını (referanslar/getup_keyframe + hasat) FAZ olarak
# takip eden bir taklit-BONUSU eklenir (ilerleme-ağırlıklı: yatık-sabit ~0,
# kalkma-yolunda güçlü pozitif). El-gradyanının ÜSTÜNE eklenir, onu ezmez.
# VARSAYILAN KAPALI — mevcut A eğitimi/izle scriptleri etkilenmez; ancak
# imitasyon_baslat.py bunu açıp yeni eğitim başlatır.
USE_IMITATION = False
IMIT_GETUP_WEIGHT = 1.0         # taklit-bonusu ağırlığı (el-gradyanına eklenir)
# v2c/ep1100 teşhisi + kullanıcı gözlemi: warm-start politika "rakip YAKIN → dur"
# davranışını taşıyor; yan yana yakın spawn'da (1.4m) bu hemen tetikleniyordu →
# yürümüyorlardı. Çözüm: rakibi ÇOK UZAĞA koy. Yüz yüze ama 15m arayla → rakip
# "uzak yürüme hedefi" rolüne geçer, warm-start'ın "rakibe doğru yürü" davranışı
# (ep750'de gait üreten) devreye girer, asla varamadığı için hiç DURMAZ → sürekli
# yürür. Rakip donmuş kopya olsa bile sorun yok: sadece uzak hedef işlevi görür.
# NOT: 15m denendi ama ÇOK UZAK = dağılım-dışı (OOD) çıktı — politika en fazla
# ~3.3m mesafede eğitildiği için 8m+ mesafede kafası karışıp geriye gidiyordu.
# Test (ep1000): 4m→+0.36m ileri %64 dik, 6m→+0.40m %39, 8m→geri, 15m→geri.
# Tatlı nokta ~4-5m: dağılım-içi (ileri yürür, dik kalır) AMA episode içinde
# yakınlaşma-durma tetiğine (~1m) ulaşacak kadar hızlı değil → sürekli yürür.
WALK_SPAWN_DIST = 5.0           # locomotion: yüz yüze başlangıç mesafesi (uzak hedef)
# v2c/ep1050 teşhisi (kullanıcı isteğiyle düzeltme): erken bitişi TAMAMEN
# kaldırınca düşen ajan episode sonuna kadar (2800 adım) yerde kalıp ceza
# biriktiriyor (-3/adım) → yürüme sinyali boğuluyordu (ep1100: %7 dik, μ=-0.38).
# Kalkma hedefi KORUNUYOR ama iki tavan ekleniyor:
#  1) yerdeki ceza sabitleniyor (birikmesin) — GROUND_PENALTY_CAP
#  2) sürekli bu kadar adım yerde kalınırsa episode biter (kalkma penceresi)
GROUND_PENALTY_CAP = -1.5       # locomotion: yerde sabit ceza (eski -2..-4 birikimi yerine)
LOCOMOTION_FALL_TERMINATE_STEPS = 250  # sürekli bu kadar adım yerdeyse episode biter
                                       # (bu süre içinde kalkarsa sayaç sıfırlanır → kalkma şansı)
# v2c/ep1200→1300 regresyon teşhisi (sigma/self-play/KL elendikten sonra kalan
# hipotez): locomotion episode'ları çok UZUN (3200 adım) → uzun ufukta değer
# (critic) tahmini zorlaşıyor → yanlış avantajlar → PPO politikayı daha DÜŞÜK
# ödüle sürüklüyor (%75→%21 dik). Kısa episode = kolay/doğru değer tahmini =
# kararlı öğrenme. 600 adım dengeli yürüme göstermeye yeter.
LOCOMOTION_MAX_STEPS = 600      # locomotion: episode üst sınırı (curriculum'u ezer)

# Early termination: bu kadar ardışık adım yerde kalan taraf maçı kaybeder.
# (180 adım x 0.003 s = 0.54 s). Düşmenin episode'u bitirmemesi, humanoid
# denge öğreniminin bilinen kırılma noktasıdır — yerde yatarak ceza
# biriktirmek yerine episode biter ve gelecekteki tüm ödül kaybedilir.
# 100 (0.3s) yaklaşırken/vururken oluşan geçici dengesizliği toparlanma
# şansı vermeden "düşme" sayıyordu (ep550 kanıtı: mesafe 1.0m→0.55m'ye
# inip tam temas eşiğinde düşüyorlardı) — 180'e çıkarıldı.
FALL_TERMINATE_STEPS = 180

class RunningMeanStd:
    """Observation normalization için running statistics"""
    def __init__(self, shape):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = 1e-4

    def update(self, x):
        """Batch update"""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        self.mean += delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        self.var = M2 / total_count
        self.count = total_count

    def normalize(self, x):
        """Normalize observation"""
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)

# ============================================
# REWARD SYSTEM
# ============================================
class ImprovedRewardSystem:
    """Gelişmiş ödül hesaplama sistemi"""

    def __init__(self):
        # FIX 6: combo_counter stats tracking için tutuldu (obs'da görünüyor)
        # ama update_combo() kaldırıldı — reward hesabında kullanılmıyor
        self.combo_counter = {'p1': 0, 'p2': 0}
        self.critical_threshold = 200.0
        self.damage_multipliers = {
            'hand': 1.0,
            'foot': 1.5,
            'head': 2.0,
            'knee': 1.8,
            'elbow': 1.6
        }

    def calculate_damage(self, force_magnitude, body_part='hand'):
        base_damage = min(force_magnitude / 80.0, 8.0)
        multiplier = self.damage_multipliers.get(body_part, 1.0)
        is_critical = force_magnitude > self.critical_threshold
        if is_critical:
            multiplier *= 1.5
        final_damage = base_damage * multiplier
        return final_damage, is_critical

    # FIX 6: update_combo() kaldırıldı.
    # hit_reward = damage * 2.0 yeterli ve daha öngörülebilir.
    def increment_combo(self, player):
        """Sadece stats tracking için sayacı artır."""
        self.combo_counter[player] += 1

    # check_strategic_position / calculate_approach_reward /
    # calculate_hp_advantage_reward kaldırıldı — reward shaping basitleştirildi.

# ============================================
# ENVIRONMENT
# ============================================
class DualHumanoidFightEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}

    def __init__(self, render_mode=None, xml_path="dual_humanoid_fight.xml", env_id=0, difficulty=1.0):
        super().__init__()
        self.combat_weight = 1.0
        # env_id'yi hemen set et (çünkü _find_body_ids'de kullanılacak)
        self.env_id = env_id

        self._ground_timers = {}
        self._fall_counter = {'p1': 0, 'p2': 0}
        # TAKLİT: kalkma faz-takipçisi (yalnız USE_IMITATION iken kurulur; P1'e özel)
        self._getup_tracker = None
        self._getup_prev_progress = 0.0
        self._getup_ref_back = None
        self._getup_ref_front = None
        if USE_IMITATION:
            try:
                import imitasyon
                _lib = imitasyon.ReferenceLibrary()
                # DÜŞME-YÖNÜNE göre iki referans: sırtüstü (arkadan) / yüzüstü (önden)
                self._getup_ref_back = _lib.getup_reference(on_back=True)
                self._getup_ref_front = _lib.getup_reference(on_back=False)
                self._getup_tracker = imitasyon.GetupTracker(self._getup_ref_back)
                if env_id == 0:
                    _b = 'YOK' if self._getup_ref_back is None else f"{len(self._getup_ref_back)} kare"
                    _f = 'YOK' if self._getup_ref_front is None else f"{len(self._getup_ref_front)} kare"
                    print(f"   TAKLİT açık: arkadan={_b}  önden={_f}")
            except Exception as _e:
                if env_id == 0:
                    print(f"   ! TAKLİT kurulamadı: {_e}")
                self._getup_tracker = None
        if env_id == 0:
            print("\n🥊 IMPROVED DUAL HUMANOID FIGHT ENV")
            print("="*60)

        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"XML dosyası bulunamadı: {xml_path}")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        # 2026-10-25 (kullanıcı: motor fiziği adım adım): robot menteşe eklemlerinin rotor ataleti ARMATUR_K, eklem sürtünmesi
        # (sönüm) SONUM_K ile AYRI ayrı ölçeklenir (varsayılan 1 = eski davranış). PD fren terimi: getup_train.PD_KD_K.
        _ak, _sk = float(os.environ.get("ARMATUR_K", "1")), float(os.environ.get("SONUM_K", "1"))
        if _ak != 1.0 or _sk != 1.0:
            for j in range(self.model.njnt):
                ad = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
                if ad.startswith(("p1_", "p2_")) and self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
                    v = self.model.jnt_dofadr[j]
                    self.model.dof_armature[v] *= _ak; self.model.dof_damping[v] *= _sk
        self.data = mujoco.MjData(self.model)
        self.difficulty = difficulty
        # v2b/ep1950: mesafe 3.0-5.0m'ye çıkarılmıştı (yürüme zorunlu kılmak
        # için). v2b/ep2900: 250+ episode boyunca (güçlendirilmiş mesafe
        # cezasına -0.20- rağmen) mesafe ~2m'lik bir platoda sıkıştı, hiç
        # yumruk menziline (0.8m) yaklaşamadılar — 3-5m çok büyük bir
        # zorluk sıçramasıydı. Aralık küçültüldü: 2.0m (taban) - 3.5m
        # (tavan). Hâlâ yürüme gerektiriyor ama mevcut isabet becerisine
        # (yakın mesafede öğrenilmiş) daha yakın bir ara hedef.
        # v2c: yürüme sıfırdan öğrenilecek → YAKINDAN başla. difficulty arttıkça
        # tavana kadar aç. Kısa yürüme mesafesi, dominant-dik + yürüyüş-alternasyon
        # ödüllerinin önce dengeli stepping'i kurmasına izin verir.
        # v2c/ep450 kullanıcı isteği: mesafe çok kısaydı (görünür yürüme yoktu),
        # uzatıldı. Ajanlar ep450'de 1.25m'yi DİK gelerek (z=0.736) kapatıyordu.
        # v2c/ep500 kullanıcı isteği: mesafeyi BAYA aç — yürümeyi öğrenmeleri için
        # yeterli alan olsun (kısa mesafede sürekli locomotion zorlaması yok).
        # v2c'nin denge-öncelikli ödülü sayesinde uzun mesafe v2b'deki gibi
        # tehlikeli değil: ulaşamasalar bile dik durunca +1.0 alıyorlar (çökmeye
        # itilmiyorlar), ileri ödül yürümeye teşvik ediyor. Taban 2.5, tavan 4.0,
        # eğim 1.5. difficulty 1.38'de ~3.07m, difficulty 2.0'de 4.0m (tavan).
        self.initial_distance = min(2.5 + (difficulty - 1.0) * 1.5, 4.0)

        self._find_body_ids()

        total_actuators = self.p1_actuator_count + self.p2_actuator_count
        self.action_space = spaces.Box(

            low=-1.0,
            high=1.0,
            shape=(total_actuators,),
            dtype=np.float32
        )
        self.player1_hp = 60.0
        self.player2_hp = 60.0
        self.timestep = 0
        self.max_timesteps = 600
        self.last_contact_time = {'p1': 0, 'p2': 0}
        self.last_damage_step = 0
        self._ground_timers = {}
        self._fall_counter = {'p1': 0, 'p2': 0}
        self._last_stance_foot = {'p1': None, 'p2': None}  # v2c yürüyüş alternasyonu takibi
        self.reward_system = ImprovedRewardSystem()
        self.stats = {
            'p1_total_damage': 0.0, 'p2_total_damage': 0.0,
            'p1_hits': 0, 'p2_hits': 0,
            'p1_criticals': 0, 'p2_criticals': 0,
            'p1_max_combo': 0, 'p2_max_combo': 0
        }
        # FIX 4: obs_size'ı _get_obs() ile otomatik ölç
        # Obs concat'e yeni satır eklendiğinde bu satırı güncellemek gerekmez.
        mujoco.mj_forward(self.model, self.data)
        obs_size = self._get_obs('p1').shape[0]

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32
        )

        if env_id == 0:
            print(f"✓ Action Space: {self.action_space.shape}")
            print(f"✓ Observation Space: {self.observation_space.shape}")
            print(f"✓ obs_size: {obs_size} (auto-measured | qpos={self.per_player_qpos_dim}, qvel={self.per_player_vel_dim})")
            print(f"✓ Difficulty: {difficulty:.2f}")
            print(f"✓ Device: {DEVICE}")
            print("="*60 + "\n")

        self.render_mode = render_mode
        self.viewer = None

        self.reward_system = ImprovedRewardSystem()

        self.player1_hp = 60.0
        self.player2_hp = 60.0
        self.timestep = 0
        self.max_timesteps = 4000  # curriculum ile güncellenir — başlangıç değeri

        self.last_contact_time = {'p1': 0, 'p2': 0}
        self.contact_cooldown = 10
        self.last_damage_step = 0

        self.stats = {
            'p1_total_damage': 0.0,
            'p2_total_damage': 0.0,
            'p1_hits': 0,
            'p2_hits': 0,
            'p1_criticals': 0,
            'p2_criticals': 0,
            'p1_max_combo': 0,
            'p2_max_combo': 0
        }

    def _find_body_ids(self):
        self.p1_body_ids   = []
        self.p2_body_ids   = []
        self.p1_body_parts = {}
        self.p2_body_parts = {}
        self.key_body_ids  = {}

        # FIX 5: Torso ID'lerini cache'le — her step O(nbody) döngüsünden kaçın
        self.p1_head_id   = None
        self.p2_head_id   = None
        self.p1_torso_id = None
        self.p2_torso_id = None
        key_bodies = [
            'p1_foot_left',  'p1_foot_right',
            'p1_shin_left',  'p1_shin_right',
            'p1_thigh_left', 'p1_thigh_right',   # FIX 7: uyluk eklendi
            'p1_fist_right', 'p1_fist_left',
            'p1_head',                             # FIX 3: kafa eklendi
            'p2_foot_left',  'p2_foot_right',
            'p2_shin_left',  'p2_shin_right',
            'p2_thigh_left', 'p2_thigh_right',
            'p2_fist_right', 'p2_fist_left',
            'p2_head',
        ]
        for i in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if not name:
                continue
            if name.startswith('p1_'):
                self.p1_body_ids.append(i)
                self.p1_body_parts[i] = self._determine_body_part(name)
            elif name.startswith('p2_'):
                self.p2_body_ids.append(i)
                self.p2_body_parts[i] = self._determine_body_part(name)
            if name in key_bodies:
                self.key_body_ids[name] = i

            if name == 'p1_torso':
                 self.p1_torso_id = i
            elif name == 'p2_torso':
                 self.p2_torso_id = i


            if name == 'p1_head':
                self.p1_head_id = i
            elif name == 'p2_head':
                self.p2_head_id = i

        # FIX 3: Per-player qpos / qvel aralıklarını cache'le
        # (global state leak'ini önler — her oyuncu sadece kendi joint'lerini görür)
        def _root_jnt(torso_id):
            return self.model.body_jntadr[torso_id] if torso_id is not None else -1

        j1 = _root_jnt(self.p1_torso_id)
        j2 = _root_jnt(self.p2_torso_id)

        if j1 >= 0 and j2 >= 0:
            self.p1_qpos_start      = int(self.model.jnt_qposadr[j1])
            self.p2_qpos_start      = int(self.model.jnt_qposadr[j2])
            self.p1_vel_start       = int(self.model.jnt_dofadr[j1])
            self.p2_vel_start       = int(self.model.jnt_dofadr[j2])
            self.per_player_qpos_dim = self.p2_qpos_start - self.p1_qpos_start
            self.per_player_vel_dim  = self.p2_vel_start  - self.p1_vel_start
        else:
            # Fallback: tüm state (güvensiz, sadece hata durumu)
            self.p1_qpos_start = self.p2_qpos_start = 0
            self.p1_vel_start  = self.p2_vel_start  = 0
            self.per_player_qpos_dim = self.model.nq
            self.per_player_vel_dim  = self.model.nv

        # FIX 5: Root DOF adresi — hız sorgusunda döngü yok
        self.p1_root_dof_adr = int(self.model.jnt_dofadr[j1]) if j1 >= 0 else -1
        self.p2_root_dof_adr = int(self.model.jnt_dofadr[j2]) if j2 >= 0 else -1

        self.p1_actuator_count = 0
        self.p2_actuator_count = 0
        for i in range(self.model.nu):
            aname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if aname and aname.startswith('p1_'):
                self.p1_actuator_count += 1
            elif aname and aname.startswith('p2_'):
                self.p2_actuator_count += 1

        if self.env_id == 0:
            print(f"✓ P1: {len(self.p1_body_ids)} bodies, {self.p1_actuator_count} actuators")
            print(f"✓ P2: {len(self.p2_body_ids)} bodies, {self.p2_actuator_count} actuators")
            print(f"✓ per_player qpos={self.per_player_qpos_dim}, qvel={self.per_player_vel_dim}")

    def _determine_body_part(self, name):
        n = name.lower()
        if 'fist' in n or 'hand' in n or 'palm' in n:
            return 'hand'
        elif 'foot' in n or 'ankle' in n:
            return 'foot'
        elif 'head' in n or 'skull' in n:   # FIX 3: p1_head / p2_head doğru tanımlanıyor
            return 'head'
        elif 'forearm' in n:
            return 'elbow'
        elif 'shin' in n or 'knee' in n:
            return 'knee'
        elif 'thigh' in n:
            return 'thigh'
        else:
            return 'body'

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._ground_timers = {}
        # TAKLİT: her episode başında kalkma-fazını sıfırla
        if self._getup_tracker is not None:
            self._getup_tracker.reset()
            self._getup_prev_progress = 0.0
        mujoco.mj_resetData(self.model, self.data)

        distance = self.initial_distance

        # FIX 1c + 5: cache'lenmiş qpos indekslerini kullan — O(nbody) döngüsü yok
        # Torso z=0.62: bacak zinciri 0.16+0.22+0.20+0.02 = 0.60 → ayaklar yere basar.
        # (Eski 0.85 değeri ajanları 23 cm havada başlatıp düşürüyordu.)
        #
        # Yerleşim X EKSENİ boyunca ve YÜZ YÜZE:
        # Omuz/kalça eklemleri y-ekseni menteşesi → yumruk/tekme +x yönüne savrulur.
        # Modelde yaw eklemi yok (dönemezler), bu yüzden spawn'da rakibe dönük
        # olmaları şart: P1 kimlik quaterniyonu ile +x'e, P2 z-ekseninde 180°
        # çevrilerek -x'e bakar. (Eski kod y ekseni boyunca YAN DÖNÜK başlatıyordu.)
        # Domain randomization: küçük bir yarıçap içinde rastgele pozisyon/açı
        # sapması. Amaç: model HER episode'da birebir aynı başlangıç
        # durumunu görmesin — aşırı-spesifik ("mean" davranışı bu tek
        # duruma aşırı uyumlu, farklı durumlarda kararsız) bir politikaya
        # kilitlenmek yerine genelleyen bir strateji öğrensin. Yarıçap
        # küçük tutuldu (±15cm, ±~9°) — checkpoint'i bozmaz, sadece reset()
        # davranışını çeşitlendirir.
        pos_r = 0.15
        ang_r = 0.15  # radyan, ~8.6°
        p1_dx, p1_dy = np.random.uniform(-pos_r, pos_r, 2)
        p2_dx, p2_dy = np.random.uniform(-pos_r, pos_r, 2)
        p1_yaw = np.random.uniform(-ang_r, ang_r)
        p2_yaw = np.pi + np.random.uniform(-ang_r, ang_r)

        # Locomotion: YÜZ YÜZE ama ÇOK UZAK (WALK_SPAWN_DIST). Rakip 'uzak
        # yürüme hedefi' rolünde — warm-start 'rakibe yürü' davranışı devreye
        # girer, asla varamaz → hiç durmaz. Dövüş: normal yakın mesafe.
        # Locomotion: uzak (WALK_SPAWN_DIST). Dövüş: normal (distance) VEYA
        # yakın-mesafe müfredatı için CLOSE_SPAWN_DIST (temas menzili).
        if LOCOMOTION_PHASE:
            spawn_dist = WALK_SPAWN_DIST
        elif CLOSE_SPAWN_DIST is not None:
            spawn_dist = CLOSE_SPAWN_DIST
        else:
            spawn_dist = distance

        if self.p1_qpos_start is not None:
            self.data.qpos[self.p1_qpos_start:self.p1_qpos_start+3] = \
                [-spawn_dist/2 + p1_dx, 0.0 + p1_dy, 0.62]
            self.data.qpos[self.p1_qpos_start+3:self.p1_qpos_start+7] = \
                [np.cos(p1_yaw/2), 0, 0, np.sin(p1_yaw/2)]
        if self.p2_qpos_start is not None:
            self.data.qpos[self.p2_qpos_start:self.p2_qpos_start+3] = \
                [ spawn_dist/2 + p2_dx, 0.0 + p2_dy, 0.62]
            self.data.qpos[self.p2_qpos_start+3:self.p2_qpos_start+7] = \
                [np.cos(p2_yaw/2), 0, 0, np.sin(p2_yaw/2)]

        self.player1_hp = 60.0
        self.player2_hp = 60.0
        self.timestep = 0
        self.last_contact_time = {'p1': 0, 'p2': 0}
        self.last_damage_step = 0
        self._fall_counter = {'p1': 0, 'p2': 0}
        self._last_stance_foot = {'p1': None, 'p2': None}  # v2c yürüyüş alternasyonu takibi

        self.reward_system = ImprovedRewardSystem()

        self.stats = {
            'p1_total_damage': 0.0,
            'p2_total_damage': 0.0,
            'p1_hits': 0,
            'p2_hits': 0,
            'p1_criticals': 0,
            'p2_criticals': 0,
            'p1_max_combo': 0,
            'p2_max_combo': 0
        }

        # GETUP: normal (dik) spawn'ı EZ → P1 DÜŞMÜŞ başlasın (yatık, alçak z).
        # Kök ~90° (±23°) yan/öne yatırılır (gövde yatay), z≈0.22, hızlar sıfır.
        # P2 uzağa (6m, alakasız). Ajan buradan KALKMAYI öğrenir. (Normal reset
        # finalizasyonundan SONRA, mj_forward'dan ÖNCE — hiçbir kurulum atlanmaz.)
        if GETUP_PHASE and self.p1_qpos_start is not None and not GETUP_SPAWN_STANDING:
            # ESKİ (yatık-spawn müfredatı) — GETUP_SPAWN_STANDING=False iken.
            ax = int(np.random.choice([0, 1]))                 # y ya da x ekseni
            ang = np.random.uniform(0.35, np.pi/2 + 0.25)      # ~20° (hafif) → ~103° (yatık)
            frac = ang / (np.pi/2)
            z0 = float(np.clip(0.58 - 0.36 * frac, 0.20, 0.58))
            half = ang / 2.0
            q = [np.cos(half), 0.0, 0.0, 0.0]
            q[1 + ax] = np.sin(half)
            self.data.qpos[self.p1_qpos_start:self.p1_qpos_start+3] = [0.0, 0.0, z0]
            self.data.qpos[self.p1_qpos_start+3:self.p1_qpos_start+7] = q
            if self.p1_root_dof_adr >= 0:
                self.data.qvel[self.p1_root_dof_adr:self.p1_root_dof_adr+6] = 0.0
            if self.p2_qpos_start is not None:
                self.data.qpos[self.p2_qpos_start:self.p2_qpos_start+3] = [6.0, 0.0, 0.62]

        elif GETUP_PHASE and self.p1_qpos_start is not None and GETUP_SPAWN_STANDING:
            # YENİ (kullanıcı fikri): DİK başla + rastgele İTME → doğal düşüş/toparlanma.
            self.data.qpos[self.p1_qpos_start:self.p1_qpos_start+3] = [0.0, 0.0, 0.62]
            self.data.qpos[self.p1_qpos_start+3:self.p1_qpos_start+7] = [1.0, 0.0, 0.0, 0.0]
            self.data.qpos[self.p1_qpos_start+7:self.p1_qpos_start+19] = 0.0   # eklemler nötr (dik)
            if self.p1_root_dof_adr >= 0:
                self.data.qvel[self.p1_root_dof_adr:self.p1_root_dof_adr+6] = 0.0
                push = float(np.random.uniform(0.0, GETUP_PUSH_MAX))
                ang = float(np.random.uniform(0, 2*np.pi))
                self.data.qvel[self.p1_root_dof_adr+0] = push*np.cos(ang)      # vx
                self.data.qvel[self.p1_root_dof_adr+1] = push*np.sin(ang)      # vy
                self.data.qvel[self.p1_root_dof_adr+3] = float(np.random.uniform(-1,1))*push*0.4  # devrilme
                self.data.qvel[self.p1_root_dof_adr+4] = float(np.random.uniform(-1,1))*push*0.4
            if self.p2_qpos_start is not None:
                self.data.qpos[self.p2_qpos_start:self.p2_qpos_start+3] = [6.0, 0.0, 0.62]

        mujoco.mj_forward(self.model, self.data)

        # ✅ FIX: Gymnasium standardı — obs, info döndür (None değil)
        obs_p1, _ = self.get_obs_both()
        return obs_p1, {}
    # _find_root_qpos_index kaldırıldı — reset() artık p1_qpos_start cache'ini kullanıyor

    def step(self, actions):
        self.timestep += 1
        action_p1 = np.clip(actions[:self.p1_actuator_count], -1, 1)
        action_p2 = np.clip(actions[self.p1_actuator_count:], -1, 1)

        self.data.ctrl[:self.p1_actuator_count] = action_p1
        self.data.ctrl[self.p1_actuator_count:self.p1_actuator_count + self.p2_actuator_count] = action_p2

        mujoco.mj_step(self.model, self.data)

        # Denge ödülü _compute_improved_rewards() içinde bir kez ekleniyor;
        # buradaki ikinci hesap (combat_weight blend) kaldırıldı — weight=1'de
        # etkisizdi ve _ground_timers'ı adım başına iki kez mutate ediyordu.
        reward_p1, reward_p2, contact_info, p1_pos, p1_vel, p2_pos, p2_vel = self._compute_improved_rewards()

        terminated = False
        truncated  = False
        winner = None

        # Early termination sayaçları: kafa FLOOR altındaysa artar, değilse sıfırlanır
        self._fall_counter['p1'] = self._fall_counter['p1'] + 1 if p1_pos[2] < HEAD_FLOOR_Z else 0
        self._fall_counter['p2'] = self._fall_counter['p2'] + 1 if p2_pos[2] < HEAD_FLOOR_Z else 0
        p1_down = self._fall_counter['p1'] >= FALL_TERMINATE_STEPS
        p2_down = self._fall_counter['p2'] >= FALL_TERMINATE_STEPS

        if GETUP_PHASE:
            # AYAĞA-KALKMA: ajan zaten DÜŞMÜŞ başlar → düşme ile bitiş YOK
            # (tüm episode boyunca kalkmayı denesin). Sadece timeout.
            if self.timestep >= self.max_timesteps:
                truncated = True
                winner = "Getup timeout"
        elif LOCOMOTION_PHASE:
            # Saf yürüme fazı: düşme/HP ile ANINDA bitiş YOK — düştüklerinde
            # KALKMA PENCERESİ verilir (bu süre içinde kalkarsa _fall_counter
            # sıfırlanır ve devam eder). Ama sürekli LOCOMOTION_FALL_TERMINATE_
            # STEPS adım yerde kalınırsa episode biter (yerde ceza birikmesini
            # ve yürüme sinyalinin boğulmasını sınırlar).
            p1_stuck = self._fall_counter['p1'] >= LOCOMOTION_FALL_TERMINATE_STEPS
            p2_stuck = self._fall_counter['p2'] >= LOCOMOTION_FALL_TERMINATE_STEPS
            if p1_stuck or p2_stuck:
                terminated = True
                winner = "Locomotion fall (kalkamadi)"
            elif self.timestep >= self.max_timesteps:
                truncated = True
                winner = "Locomotion timeout"
        elif self.player1_hp <= 0:
            terminated = True
            winner = "Player 2 (Blue) - Knockout!"
            reward_p1 -= 100
            reward_p2 += 100
        elif self.player2_hp <= 0:
            terminated = True
            winner = "Player 1 (Red) - Knockout!"
            reward_p1 += 100
            reward_p2 -= 100
        elif p1_down and p2_down and not COMBINE_PHASE:
            terminated = True
            winner = "Draw (both down)"
            reward_p1 -= 30
            reward_p2 -= 30
        elif p1_down and not COMBINE_PHASE:   # COMBINE: düşme BİTİRMEZ (kalksın)
            terminated = True
            # "Hak edilmiş" mi? Rakip (p2) yakın zamanda isabet ettirdiyse
            # gerçek knockdown; değilse p1 kendi kendine düştü (keşif
            # gürültüsü/denge kaybı) — p2'ye kazanma ödülü VERİLMEZ.
            # v3-yeni-mimari kanıtı: gürültülü 6 maçta 3 temas ama 6 knockdown
            # (3-3) — rakibe hiç dokunmadan "kazanmak" mümkündü, ödül isabete
            # değil ayakta-kalma yarışına akıyordu.
            earned = (self.timestep - self.last_contact_time['p2']) < 60
            # "Draw" ile başlamak bilinçli: win-rate sayaçları winner string'inde
            # "Player 1"/"Player 2" arıyor — hak edilmemiş düşmeyi galibiyet
            # olarak SAYMAMASI için "Draw" ile başlıyor (nötr, kazanan yok).
            # v3: ±25 çok ağır bir tek-seferlik risk oldu — ajan bunu göze
            # almaktansa mesafe cezasını göze alıp kaçmayı tercih etti.
            # ±12'ye indirildi (mesafe cezası artışıyla dengelenir).
            winner = "Player 2 (Blue) - Knockdown!" if earned else "Draw (P1 fell, no strike)"
            reward_p1 -= 12
            reward_p2 += 12 if earned else 0
        elif p2_down and not COMBINE_PHASE:   # COMBINE: düşme BİTİRMEZ
            terminated = True
            earned = (self.timestep - self.last_contact_time['p1']) < 60
            winner = "Player 1 (Red) - Knockdown!" if earned else "Draw (P2 fell, no strike)"
            reward_p2 -= 12
            reward_p1 += 12 if earned else 0
        elif self.timestep >= self.max_timesteps:
            # ✅ FIX: Zaman dolması terminated değil, truncated olmalı
            truncated = True
            if self.player1_hp == self.player2_hp:
                winner = f"Draw (Tie at {self.player1_hp:.0f} HP each)"
            elif self.player1_hp > self.player2_hp:
                winner = f"Player 1 (Red) - Decision ({self.player1_hp:.0f} vs {self.player2_hp:.0f} HP)"
                reward_p1 += 50
                reward_p2 -= 50
            else:
                winner = f"Player 2 (Blue) - Decision ({self.player2_hp:.0f} vs {self.player1_hp:.0f} HP)"
                reward_p1 -= 50
                reward_p2 += 50

        info = {
            'p1_hp':         self.player1_hp,
            'p2_hp':         self.player2_hp,
            'contact_info':  contact_info,
            'winner':        winner,
            'stats':         self.stats
        }

        # ✅ Gymnasium standardı: 5 değer (obs, reward, terminated, truncated, info)
        # reward sözlük olarak döndürülür — ödül ölçeklerini korur.
        obs_p1, obs_p2 = self.get_obs_both()
        obs    = (obs_p1, obs_p2)
        reward = {'p1': reward_p1, 'p2': reward_p2}
        return obs, reward, terminated, truncated, info
    def get_obs_both(self):
        obs_p1 = self._get_obs('p1')
        obs_p2 = self._get_obs('p2')
        return obs_p1, obs_p2
    def _compute_balance_reward(self, player_pos, player_vel, player_key, timestep,
                                torso_upright=1.0):
        """
        v2c denge ödülü — dik/sağlıklı bonusu BASKIN sinyal (Humanoid-v4 tarzı):
          z >= UPRIGHT_Z ve gövde dik : +UPRIGHT_HEALTHY_BONUS (+1.0)  ← baskın
          z >= UPRIGHT_Z ama eğik     : +0.3  (yüksek ama dengesiz — az ödül)
          HEAD_FLOOR_Z ≤ z            : doğrusal ceza  →  0 .. -1.5
          z < HEAD_FLOOR_Z            : yerde kalma cezası → -2.0 .. -4.0
        v2b'de dik bonus sadece +0.3'tü — mesafe cezası (~-0.5/adım) bunu ezip
        politikayı dengeyi feda edip atılmaya itiyordu. Artık dik+dengeli kalmak
        en büyük tek-adım ödülü; ileri hareket ödülü (ayrı, sınırlı) bunun ÜSTÜNE
        eklenir, böylece "dengeni koru, sonra yürü" hiyerarşisi kurulur.
        """
        z           = float(player_pos[2]) if len(player_pos) > 2 else 1.0
        ground_key  = f'{player_key}_ground_since'

        if z >= HEAD_UPRIGHT_Z:
            self._ground_timers.pop(ground_key, None)
            # TAKLİT: ayağa kalkınca (dik) kalkma-fazını sıfırla → SONRAKİ düşüş
            # baştan (yatık→dik) izlenir; monoton indeks takılı kalmaz.
            if USE_IMITATION and player_key == 'p1' and self._getup_tracker is not None:
                self._getup_tracker.reset()
                self._getup_prev_progress = 0.0
            # COMBINE: yüksekteyken bile FIRLIYORSA (hızlı) upright bonusunu
            # FARMLAMASIN → anti-fırlama cezası (fırlama açığını kapatır).
            if COMBINE_PHASE and abs(float(player_vel[2]) if len(player_vel) > 2 else 0.0) > 2.5:
                return -0.3 * abs(float(player_vel[2]))
            if torso_upright > TORSO_UPRIGHT_MIN:
                return UPRIGHT_HEALTHY_BONUS                    # dik + dengeli (baskın)
            return 0.3                                         # yüksek ama eğik

        elif COMBINE_PHASE:
            # BİRLEŞİK FAZ getup-gradyanı: dik-altı → yerden yükselmeyi ödüllendir
            # (gövde-diklik + baş-yüksek), ANTI-FIRLAMA (dikey-hız cezası, pozitif
            # sadece sakinken). Ayakta bonusuyla SÜREKLİ: dik eşiğinde ~+1.0'a
            # ulaşır (UPRIGHT_HEALTHY_BONUS ile eşleşir), yerde ~-0.2 (kalkmaya it).
            # GÜÇLENDİRİLDİ (2026-07-16): 800-step'te ajan düşünce YERDE KALMAYI
            # seçiyordu (kalkmak zor). Yerde cezası -0.2 → -0.8 (4x güçlü). Yerde
            # -0.8, dik +1.0 (1.8 fark = güçlü kalkma itkisi). ANTI-FIRLAMA
            # korunuyor (vz>2.5 → pozitif YOK, sadece ceza) — güçlü ceza
            # fırlamayı tetiklemesin.
            self._ground_timers.pop(ground_key, None)
            vz = abs(float(player_vel[2])) if len(player_vel) > 2 else 0.0
            prog = 0.5 * max(torso_upright, 0.0) + 0.5 * min(z / HEAD_UPRIGHT_Z, 1.0)
            base = (1.8 * prog - 0.8) if vz < 2.5 else (-0.3 * vz)  # yerde -0.8→dik+1.0 / fırlama ceza
            # TAKLİT-BONUSU (yalnız P1): kalkma-referansı boyunca ilerlemeyi ödüllendir.
            # İlerleme-ağırlıklı → yatık-sabit ~0 (el-gradyanı zaten -0.8 ceza verir),
            # kalkma-yolunda güçlü pozitif. El-gradyanının ÜSTÜNE eklenir.
            if USE_IMITATION and player_key == 'p1' and self._getup_tracker is not None \
                    and self._getup_tracker.available():
                import imitasyon
                # YENİ düşüş (faz başında): düşme-yönünü tespit et → doğru referans.
                # torso xmat[6] < 0 → sırtüstü (arkadan); > 0 → yüzüstü (önden).
                if self._getup_tracker.idx == 0 and self._getup_ref_front is not None:
                    on_back = float(self.data.xmat[self.p1_torso_id][6]) < 0.0
                    ref = self._getup_ref_back if on_back else self._getup_ref_front
                    if ref is not None:
                        self._getup_tracker.set_reference(ref)
                qp = self.data.qpos[self.p1_qpos_start:self.p1_qpos_start + 19]
                bonus, self._getup_prev_progress = imitasyon.getup_imitation_bonus(
                    self._getup_tracker, qp)      # bonus = w*benzerlik*ilerleme (+taban)
                base += IMIT_GETUP_WEIGHT * bonus
            return base

        elif z >= HEAD_FLOOR_Z:
            self._ground_timers.pop(ground_key, None)
            t = 1.0 - (z - HEAD_FLOOR_Z) / (HEAD_UPRIGHT_Z - HEAD_FLOOR_Z)  # 1=yere yakın
            return -1.5 * t

        else:
            if ground_key not in self._ground_timers:
                self._ground_timers[ground_key] = timestep
            time_on_ground = timestep - self._ground_timers[ground_key]
            if LOCOMOTION_PHASE:
                # Sabit (birikmeyen) ceza: uzun episode'da yürüme sinyalini
                # boğmasın. Kalkma teşviki yine güçlü: yerde -1.5 → dik +1.0
                # (+2.5 sıçrama), ama sınırlı → öğrenme dengeli.
                return GROUND_PENALTY_CAP
            return -2.0 - min(time_on_ground / 50.0, 2.0)      # -2.0 → -4.0 (dövüş modu)
    def _gait_alternation_bonus(self, player):
        """Tek-ayak destek DEĞİŞİMİ başına küçük ödül — gerçek adım atmayı
        (sıralı L→R→L ayak teması) teşvik eder. Statik durma (iki ayak yerde →
        destek 'yok') veya iki-ayak sürüklenme ödül ALMAZ; sadece bir ayaktan
        diğerine geçiş 'adım' sayılır. v2c'nin 'öne düşerek yaklaşma yerine
        yürü' hedefinin çekirdeği. Destek ayağı = yere basan (z < CONTACT_Z)
        ve diğeri havadayken tek başına basan ayak."""
        fl = self.key_body_ids.get(f'{player}_foot_left')
        fr = self.key_body_ids.get(f'{player}_foot_right')
        if fl is None or fr is None:
            return 0.0
        zl = float(self.data.xpos[fl][2])
        zr = float(self.data.xpos[fr][2])
        left_down  = zl < GAIT_CONTACT_Z
        right_down = zr < GAIT_CONTACT_Z
        if left_down and not right_down:
            stance = 'L'
        elif right_down and not left_down:
            stance = 'R'
        else:
            stance = None   # ikisi de yerde (durma) veya ikisi de havada (sıçrama)
        bonus = 0.0
        prev = self._last_stance_foot.get(player)
        if stance is not None and prev is not None and stance != prev:
            bonus = GAIT_ALTERNATION_BONUS
        if stance is not None:
            self._last_stance_foot[player] = stance
        return bonus
    def _compute_improved_rewards(self):
        """
        Basitleştirilmiş ödül sistemi — 4 bileşen:
          1. Savaş  : hasar ver / al  (ana sinyal)
          2. Denge  : dik dur bonus / yerde kalma cezası
          3. Yaklaşma: rakibe doğru hız (tek satır, sınırlı)
          4. Pasiflik cezası: uzun süre hasar yoksa
        """
        reward_p1 = 0.0
        reward_p2 = 0.0
        damage_event = False
        contact_info = {
            'total_contacts': 0, 'p1_hits': 0, 'p2_hits': 0,
            'p1_criticals': 0,   'p2_criticals': 0, 'max_force': 0.0
        }

        p1_pos  = self._get_player_position('p1')
        p2_pos  = self._get_player_position('p2')
        p1_vel  = self._get_player_velocity('p1')
        p2_vel  = self._get_player_velocity('p2')

        # ── AYAĞA-KALKMA (GETUP) FAZI ────────────────────────────
        # Ajan DÜŞMÜŞ başlar, dik pozisyona gelmeyi öğrenir. Dövüş/yürüme YOK.
        # YOĞUN ödül: baş yüksekliği + gövde dikliği → yerdeyken bile "yukarı her
        # ilerleme" ödüllenir (gradyan var, seyrek-ödül değil), tam kalkınca
        # baskın bonus. P2 alakasız (sıfır aksiyon). Fall-termination YOK (step).
        if GETUP_PHASE:
            # KÖK-FİX (2026-07-16, kullanıcı gözlemi): eski ödül SADECE baş-z'yi
            # ödüllendiriyordu → ajan kendini HAVAYA FIRLATIP (z=2.5m, dikey-hız
            # 5m/s) ödülü SÖMÜRDÜ, gerçekten kalkmadı (reward hacking). Yeni ödül
            # GERÇEK KONTROLLÜ AYAKTA DURMAYI gerektirir:
            #  (1) gövde dikliği BASKIN (fırlayan/takla atan gövde başarısız),
            #  (2) AYAKLAR YERDE (fırlarsa ayaklar havada — ayakta durmak ister),
            #  (3) baş ayakta ~0.9'a kadar ödül, ÜSTÜ CEZALI (fırlama = yüksek z),
            #  (4) ANTI-FIRLAMA: dikey-hız cezası,
            #  (5) hepsi+sakin → GERÇEK ayakta durma bonusu.
            # GÜÇLENDİRİLMİŞ ANTI-FIRLAMA (2026-07-16, ep100 hala vz=5 fırlıyordu):
            # Pozitif ödül (yükseklik/gövde/ayak) SADECE SAKİN durumda verilir
            # (|vz|<2.5). Fırlarken (yüksek dikey-hız) SIFIR pozitif ödül + güçlü
            # ceza → ajan ödül almak için MECBUREN sakin/kararlı olmalı. (Getup
            # itişi kısa süreli vz~1-2 olur, 2.5 eşiği buna izin verir; 5m/s
            # fırlama tamamen dışlanır.)
            torso_up = float(self.data.xmat[self.p1_torso_id][8])
            z = float(p1_pos[2])
            flz = float(self.data.xpos[self.key_body_ids['p1_foot_left']][2])
            frz = float(self.data.xpos[self.key_body_ids['p1_foot_right']][2])
            foot_z = min(flz, frz)
            vz = float(self.data.qvel[self.p1_root_dof_adr + 2]) if self.p1_root_dof_adr >= 0 else 0.0
            # KÖK-FİX-2 (2026-08, ep200 %55 fırlama): eski ödül yükseklik/gövde-dik
            # ödülünü AYAKLAR YERDE şartı OLMADAN veriyordu → ajan gövdeyi HAVADA
            # dik tutup (fırlama) ödül topladı. ARTIK tüm pozitif ödül SADECE
            # AYAKLAR YERDE (planted) + sakin iken. Havadaki gövde = fırlama =
            # pozitif YOK + ek ceza. Kalkma zaten ayaklar-yerde push ile olur.
            planted = foot_z < 0.18
            r = -0.3 * abs(vz)                                        # (4) anti-fırlama (güçlü)
            if abs(vz) < 2.5 and planted:                            # POZİTİF: SADECE ayak-yerde + sakin
                r += 2.0 * max(torso_up, 0.0)                         # (1) gövde dik (baskın)
                r += 1.5 * min(z / 0.9, 1.0)                          # (3) baş → ayakta
                r += 1.0                                              # (2) ayaklar yerde (şart)
                if torso_up > 0.7 and z > 0.6 and abs(vz) < 1.0:
                    r += 2.0                                          # (5) GERÇEK ayakta durma
            elif not planted:
                r -= 0.5                                              # ayaklar HAVADA (fırlama) → ek ceza
            act1 = self.data.ctrl[:self.p1_actuator_count]
            r -= 0.02 * float(np.sum(np.square(act1)))                # kontrol cezası
            return r, 0.0, contact_info, p1_pos, p1_vel, p2_pos, p2_vel

        # ── SAF LOCOMOTION FAZI ──────────────────────────────────
        # Dövüş/yaklaşma/mesafe/yumruk/pasiflik YOK. Sadece: dik kal (baskın) +
        # sürekli ileri yürü (+x, sınırlı) + gerçek adım (gait) + kontrol cezası.
        # Her iki ajan da +x'e bakıyor → dünya +x hızı = ego-ileri hız (v2c gait
        # buraya transfer olur). Düşünce ödül yapısı (yerde -3, dik +1) kalkmayı
        # teşvik eder; step() düşünce episode'u BİTİRMEZ, kalkma öğrenilir.
        if LOCOMOTION_PHASE:
            p1_torso = float(self.data.xmat[self.p1_torso_id][8])
            p2_torso = float(self.data.xmat[self.p2_torso_id][8])
            reward_p1 += self._compute_balance_reward(p1_pos, p1_vel, 'p1', self.timestep, p1_torso)
            reward_p2 += self._compute_balance_reward(p2_pos, p2_vel, 'p2', self.timestep, p2_torso)
            p1_up = (p1_pos[2] > HEAD_UPRIGHT_Z) and (p1_torso > TORSO_UPRIGHT_MIN)
            p2_up = (p2_pos[2] > HEAD_UPRIGHT_Z) and (p2_torso > TORSO_UPRIGHT_MIN)
            # İleri ödül = uzak hedefe (rakibe) doğru hız, sınırlı. Yüz yüze
            # durdukları için "rakibe doğru" = ego-ileri; warm-start yürümesiyle
            # uyumlu. Rakip 15m uzakta, asla varamaz → sürekli ödül, hiç durma.
            dvec = (p2_pos[:2] - p1_pos[:2]); dn = np.linalg.norm(dvec) + 1e-6
            # KÖK-NEDEN DÜZELTMESİ (2026-07-16): clip(-TARGET, +TARGET) — ileri
            # ödül (maks +W·TARGET) VE geriye CEZA (maks -W·TARGET). Eski
            # min(max(fwd,0),TARGET) geriye-hareketi 0'a kırpıp "bedava"
            # yapıyordu → geriye-yaslanma yerel-optimumu. Artık geriye sürüklenme
            # aktif olarak pahalı.
            if p1_up:
                fwd1 = float(np.dot(p1_vel[:2], dvec / dn))
                reward_p1 += FORWARD_REWARD_WEIGHT * float(np.clip(fwd1, -FORWARD_TARGET_SPEED, FORWARD_TARGET_SPEED))
                reward_p1 += self._gait_alternation_bonus('p1')
            if p2_up:
                fwd2 = float(np.dot(p2_vel[:2], -dvec / dn))
                reward_p2 += FORWARD_REWARD_WEIGHT * float(np.clip(fwd2, -FORWARD_TARGET_SPEED, FORWARD_TARGET_SPEED))
                reward_p2 += self._gait_alternation_bonus('p2')
            # kontrol cezası (aşırı/sarsıntılı tork): düzgün yürüyüşü teşvik
            act1 = self.data.ctrl[:self.p1_actuator_count]
            act2 = self.data.ctrl[self.p1_actuator_count:self.p1_actuator_count + self.p2_actuator_count]
            reward_p1 -= 0.02 * float(np.sum(np.square(act1)))
            reward_p2 -= 0.02 * float(np.sum(np.square(act2)))
            return reward_p1, reward_p2, contact_info, p1_pos, p1_vel, p2_pos, p2_vel

        distance = np.linalg.norm(p1_pos[:2] - p2_pos[:2])

        # 'head' saldırı parçası DEĞİL: eski kodda rakibin kafasına yumruk atınca
        # kafa da "saldıran" sayılıyor ve yumruğu atan (2x kafa çarpanıyla!)
        # hasar alıyordu — vurmak zarar ediyordu.
        ATTACK_PARTS = {'hand', 'foot', 'knee', 'elbow'}
        # Vuruş eşiği müfredatı: difficulty 1.0→1.5 aralığında 3→20'ye çıkar.
        # Sabit 20 iken rastgele keşifle "tam güçte isabet" bulmak çok zordu
        # (ep450: %99 draw, sıfır isabet). Düşük eşikle başlamak "temas=ödül"
        # ilişkisini önce öğretir, sonra gerçek güç gerekliliğine yumuşak geçer.
        MIN_FORCE = float(np.clip(3.0 + (self.difficulty - 1.0) * 34.0, 3.0, 20.0))
        # Vuran/vurulan ayrımı: vuran uzuv hızlı hareket eden taraftır.
        # Yaslanma/itişme temaslarını ve "blok yapan el"in hasar vermesini önler.
        STRIKE_SPEED_MIN   = 0.8   # m/s — vuran uzvun asgari hızı
        STRIKE_SPEED_RATIO = 1.3   # vuran uzuv, değdiği uzuvdan bu kat hızlı olmalı

        # ── 1. Savaş (temas döngüsü) ─────────────────────────────
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1 = self.model.geom_bodyid[contact.geom1]
            g2 = self.model.geom_bodyid[contact.geom2]
            p1_inv = g1 in self.p1_body_ids or g2 in self.p1_body_ids
            p2_inv = g1 in self.p2_body_ids or g2 in self.p2_body_ids
            if not (p1_inv and p2_inv):
                continue

            contact_info['total_contacts'] += 1
            c_arr = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, i, c_arr)
            force = np.linalg.norm(c_arr[:3])
            contact_info['max_force'] = max(contact_info['max_force'], force)
            if force < MIN_FORCE:
                continue

            p1_bid  = g1 if g1 in self.p1_body_ids else g2
            p2_bid  = g1 if g1 in self.p2_body_ids else g2
            p1_part = self.p1_body_parts.get(p1_bid, 'body')
            p2_part = self.p2_body_parts.get(p2_bid, 'body')

            # Uzuv hızları (cvel[3:6] = dünya çerçevesinde lineer hız).
            # Vuran taraf = hızlı hareket eden uzuv; ikisi de yavaşsa itişmedir.
            p1_spd = float(np.linalg.norm(self.data.cvel[p1_bid][3:6]))
            p2_spd = float(np.linalg.norm(self.data.cvel[p2_bid][3:6]))
            p1_striking = (p1_part in ATTACK_PARTS and p1_spd > STRIKE_SPEED_MIN
                           and p1_spd > STRIKE_SPEED_RATIO * p2_spd)
            p2_striking = (p2_part in ATTACK_PARTS and p2_spd > STRIKE_SPEED_MIN
                           and p2_spd > STRIKE_SPEED_RATIO * p1_spd)

            if p1_striking and (self.timestep - self.last_contact_time['p1']) > self.contact_cooldown:
                dmg, crit = self.reward_system.calculate_damage(force, p1_part)
                self.player2_hp        -= dmg
                self.last_contact_time['p1'] = self.timestep
                damage_event = True
                self.reward_system.increment_combo('p1')  # FIX 6: sadece sayaç
                r = dmg * 2.0                             # sade, öngörülebilir
                if crit:
                    r *= 1.3
                    contact_info['p1_criticals']   += 1
                    self.stats['p1_criticals']     += 1
                reward_p1 += r
                reward_p2 -= dmg          # hasar alan taraf direkt ceza
                contact_info['p1_hits']        += 1
                self.stats['p1_hits']          += 1
                self.stats['p1_total_damage']  += dmg
                self.stats['p1_max_combo'] = max(
                    self.stats['p1_max_combo'], self.reward_system.combo_counter['p1'])

            if p2_striking and (self.timestep - self.last_contact_time['p2']) > self.contact_cooldown:
                dmg, crit = self.reward_system.calculate_damage(force, p2_part)
                self.player1_hp        -= dmg
                self.last_contact_time['p2'] = self.timestep
                damage_event = True
                self.reward_system.increment_combo('p2')  # FIX 6
                r = dmg * 2.0
                if crit:
                    r *= 1.3
                    contact_info['p2_criticals']   += 1
                    self.stats['p2_criticals']     += 1
                reward_p2 += r
                reward_p1 -= dmg
                contact_info['p2_hits']        += 1
                self.stats['p2_hits']          += 1
                self.stats['p2_total_damage']  += dmg
                self.stats['p2_max_combo'] = max(
                    self.stats['p2_max_combo'], self.reward_system.combo_counter['p2'])

        # ── 2. Denge (dik/dengeli = BASKIN ödül) ─────────────────
        # Gövde dikliği (xmat[8]) ÖNCE hesaplanır: hem denge ödülü hem
        # "aktif dik" kapısı bunu kullanır. Öne eğilmeyi yükseklik düşmeden
        # ÖNCE yakalar (v2b ep3300 kanıtı: momentumla düşen ajan son ana
        # kadar z>0.60 kalabiliyordu).
        p1_torso_upright = float(self.data.xmat[self.p1_torso_id][8])
        p2_torso_upright = float(self.data.xmat[self.p2_torso_id][8])
        reward_p1 += self._compute_balance_reward(p1_pos, p1_vel, 'p1', self.timestep, p1_torso_upright)
        reward_p2 += self._compute_balance_reward(p2_pos, p2_vel, 'p2', self.timestep, p2_torso_upright)

        p1_actively_upright = (p1_pos[2] > HEAD_UPRIGHT_Z) and (p1_torso_upright > TORSO_UPRIGHT_MIN)
        p2_actively_upright = (p2_pos[2] > HEAD_UPRIGHT_Z) and (p2_torso_upright > TORSO_UPRIGHT_MIN)

        # ── 3. İleri hareket (SINIRLI — atılım avantajını yok eder) ──
        # v2c değişikliği: eski SINIRSIZ yaklaşma ödülü (dot(vel,dir)*0.7) +
        # mesafe cezası, ATILIMI ödüllendiriyordu (ileri hız ne kadar büyükse
        # o kadar ödül → 1.27 m/s atılım = maks ödül, sonra düşüş). Artık ileri
        # hız FORWARD_TARGET_SPEED'de (0.5 m/s) DOYAR: kontrollü yürüyüş ile
        # atılım AYNI ödülü alır (maks +0.5), ama atılım sonra düşüp baskın
        # dik-bonusunu (+1.0/adım) kaybettiği için net zararlı. Mesafe cezası
        # TAMAMEN KALDIRILDI — "yaklaş at all costs" baskısı, atılımın asıl
        # sebebiydi (v2b lean-trace kanıtı). Sadece dik+dengeliyken verilir.
        if distance > 1.0:
            if p1_actively_upright:
                dir1 = (p2_pos[:2] - p1_pos[:2]) / (distance + 1e-6)
                fwd1 = float(np.dot(p1_vel[:2], dir1))
                reward_p1 += FORWARD_REWARD_WEIGHT * min(max(fwd1, 0.0), FORWARD_TARGET_SPEED)
            if p2_actively_upright:
                dir2 = (p1_pos[:2] - p2_pos[:2]) / (distance + 1e-6)
                fwd2 = float(np.dot(p2_vel[:2], dir2))
                reward_p2 += FORWARD_REWARD_WEIGHT * min(max(fwd2, 0.0), FORWARD_TARGET_SPEED)

        # ── 3b. Yürüyüş alternasyonu (gerçek ADIM atmayı teşvik) ─────
        # Tek-ayak desteği L→R→L değiştikçe küçük bonus. Statik durma (iki
        # ayak yerde) veya iki-ayak sürüklenme ödül ALMAZ — sadece sıralı
        # adım. "Öne düşerek yaklaşma" yerine yürüyüşü teşvik eden anahtar.
        reward_p1 += self._gait_alternation_bonus('p1')
        reward_p2 += self._gait_alternation_bonus('p2')

        # ── 3c. Yumruk teşviki: rakibin kafasına doğru hızlanan yumruk
        # hıza orantılı küçük ödül alır (maks +0.1/adım). "Hiçbir şey →
        # temasta büyük ödül" arasındaki köprü; rastgele keşif bu iğneyi
        # kendi başına bulamıyordu (v4: %97 beraberlik, sıfır temas).
        if distance < 1.5:
            if p1_actively_upright:
                reward_p1 += self._fist_attack_bonus('p1', p2_pos)
            if p2_actively_upright:
                reward_p2 += self._fist_attack_bonus('p2', p1_pos)
        # ── 4. Pasiflik cezası (SADECE menzilde) ─────────────────
        # v2c değişikliği: eski her-yerde -0.35 ceza, ajan uzaktayken de
        # acıtıp "hızlıca yaklaş/bitir" baskısı yaratıyordu (atılıma katkı).
        # Artık sadece yumruk menzilindeyken (<1.5m) uygulanır — yaklaşmaya
        # değil, yaklaştıktan sonra VURMAYA iter. Yürüme fazında ceza yok.
        if damage_event:
            self.last_damage_step = self.timestep
        # DÖVÜŞ FAZI-2 (2026-07-16): Faz-1b'nin güçlü -0.3 cezası GERİ TEPTİ —
        # ajan vuramadığı için menzilde durmak sürekli cezalı oldu → KAÇTI
        # (2.28m) + dengesizleşti (dik %41). Ceza vuruşu değil kaçışı tetikledi
        # (çünkü vuruş erişilemezdi). ÇÖZÜM: yakın-mesafe müfredatı (CLOSE_SPAWN_
        # DIST) vuruşu ERİŞİLEBİLİR yapar; ceza tekrar GENTLE (-0.1/200) — sadece
        # hafif dürtsün, kaçışı/bozulmayı tetiklemesin. Asıl itki: yakın spawn +
        # kolay isabet + dev landing ödülü.
        if distance < 1.5 and (self.timestep - self.last_damage_step) > 200:
            reward_p1 -= 0.1
            reward_p2 -= 0.1
        # ── 5. Action smoothness cezası ─────────────────────
        # 0.10 çok sertti: 10 actuator × 1.0² × 0.10 = 1.0/adım ceza,
        # dik durma bonusunun (+0.4) 2.5 katı — hareketsizliği ödüllendiriyordu.
        action_p1 = self.data.ctrl[:self.p1_actuator_count]
        action_p2 = self.data.ctrl[self.p1_actuator_count:self.p1_actuator_count + self.p2_actuator_count]
        reward_p1 -= 0.02 * float(np.sum(np.square(action_p1)))
        reward_p2 -= 0.02 * float(np.sum(np.square(action_p2)))
        return reward_p1, reward_p2, contact_info, p1_pos, p1_vel, p2_pos, p2_vel
    def _get_obs(self, player='p1'):
        # FIX 3: Global state leak kaldırıldı.
        # Her oyuncu sadece KENDİ joint pozisyonlarını/hızlarını görür.
        if player == 'p1':
            own_qpos = self.data.qpos[self.p1_qpos_start :
                                      self.p1_qpos_start + self.per_player_qpos_dim].copy()
            own_qvel = self.data.qvel[self.p1_vel_start  :
                                      self.p1_vel_start  + self.per_player_vel_dim].copy()
        else:
            own_qpos = self.data.qpos[self.p2_qpos_start :
                                      self.p2_qpos_start + self.per_player_qpos_dim].copy()
            own_qvel = self.data.qvel[self.p2_vel_start  :
                                      self.p2_vel_start  + self.per_player_vel_dim].copy()

        if player == 'p1':
            own_hp   = self.player1_hp / 100.0
            opp_hp   = self.player2_hp / 100.0
            own_pos  = self._get_player_position('p1')
            opp_pos  = self._get_player_position('p2')
            own_vel  = self._get_player_velocity('p1')
            opp_vel  = self._get_player_velocity('p2')
            own_combo = self.reward_system.combo_counter['p1'] / 10.0
            opp_combo = self.reward_system.combo_counter['p2'] / 10.0
            own_prefix = 'p1'
            opp_prefix = 'p2'
            torso_id = self.p1_torso_id
        else:
            own_hp   = self.player2_hp / 100.0
            opp_hp   = self.player1_hp / 100.0
            own_pos  = self._get_player_position('p2')
            opp_pos  = self._get_player_position('p1')
            own_vel  = self._get_player_velocity('p2')
            opp_vel  = self._get_player_velocity('p1')
            own_combo = self.reward_system.combo_counter['p2'] / 10.0
            opp_combo = self.reward_system.combo_counter['p1'] / 10.0
            own_prefix = 'p2'
            opp_prefix = 'p1'
            torso_id = self.p2_torso_id

        # ── EGO-SANTRİK ÇERÇEVE (Grup C) ─────────────────────────
        # Her şeyi ajanın KENDİ bakış yönüne (torso yaw) döndür → rakip
        # her zaman "+x = ileride". P1 ve P2 simetrik durumu AYNI görür,
        # tek politika iki taraftan da öğrenir (veri verimi ~2x).
        xmat = self.data.xmat[torso_id]            # 3x3 rotasyon (satır-major)
        yaw  = np.arctan2(xmat[3], xmat[0])         # torso yerel x-ekseninin yaw'ı
        c, s = np.cos(yaw), np.sin(yaw)

        def rot(v):
            v = np.asarray(v, dtype=np.float32)
            return np.array([c*v[0] + s*v[1], -s*v[0] + c*v[1], v[2]], dtype=np.float32)

        # Projected gravity: gövde eğimi, yaw'dan BAĞIMSIZ (root quat yerine)
        proj_grav = np.array([-xmat[6], -xmat[7], -xmat[8]], dtype=np.float32)

        # Root'u (qpos[:7]=xyz+quat) ego-centric ile değiştir:
        # mutlak xy+yaw atılır (translasyon/rotasyon değişmezliği),
        # eklem açıları içseldir → aynen kalır.
        joint_angles = own_qpos[7:]
        root_lin = rot(own_qvel[0:3])              # kök lineer hız → yerel
        root_ang = rot(own_qvel[3:6])              # kök açısal hız → yerel
        joint_vels = own_qvel[6:]

        distance = np.linalg.norm(own_pos[:2] - opp_pos[:2])
        rel_pos  = rot(opp_pos - own_pos)          # rakip → yerel çerçeve

        def bp(name):
            bid = self.key_body_ids.get(name)
            return self.data.xpos[bid].copy() if bid is not None else np.zeros(3)

        # Tüm uzuv offsetleri torso'ya göreli + yerel çerçeveye döndürülmüş
        own_foot_l = rot(bp(f'{own_prefix}_foot_left')  - own_pos)
        own_foot_r = rot(bp(f'{own_prefix}_foot_right') - own_pos)
        own_shin_l = rot(bp(f'{own_prefix}_shin_left')  - own_pos)
        own_shin_r = rot(bp(f'{own_prefix}_shin_right') - own_pos)
        own_thigh_l = rot(bp(f'{own_prefix}_thigh_left')  - own_pos)
        own_thigh_r = rot(bp(f'{own_prefix}_thigh_right') - own_pos)
        own_fist_r  = rot(bp(f'{own_prefix}_fist_right')  - own_pos)
        own_fist_l  = rot(bp(f'{own_prefix}_fist_left')   - own_pos)

        opp_foot_l = rot(bp(f'{opp_prefix}_foot_left')  - opp_pos)
        opp_foot_r = rot(bp(f'{opp_prefix}_foot_right') - opp_pos)
        opp_shin_l = rot(bp(f'{opp_prefix}_shin_left')  - opp_pos)
        opp_shin_r = rot(bp(f'{opp_prefix}_shin_right') - opp_pos)
        opp_thigh_l = rot(bp(f'{opp_prefix}_thigh_left')  - opp_pos)
        opp_thigh_r = rot(bp(f'{opp_prefix}_thigh_right') - opp_pos)
        opp_fist_r  = rot(bp(f'{opp_prefix}_fist_right')  - opp_pos)
        opp_fist_l  = rot(bp(f'{opp_prefix}_fist_left')   - opp_pos)

        obs = np.concatenate([
            [own_pos[2] / 2.0],       # yükseklik (mutlak xy YOK)
            proj_grav,                # gövde eğimi (yaw-bağımsız)
            joint_angles,             # eklem açıları (içsel)
            root_lin / 10.0,
            root_ang / 10.0,
            joint_vels,               # eklem hızları (içsel)
            [own_hp],
            [opp_hp],
            [distance / 10.0],
            rel_pos / 10.0,
            [opp_pos[2] / 2.0],
            [own_combo],
            [opp_combo],
            [self.timestep / self.max_timesteps],
            rot(opp_vel) / 10.0,      # rakip hızı yerel çerçevede
            own_foot_l  / 2.0,
            own_foot_r  / 2.0,
            own_shin_l  / 2.0,
            own_shin_r  / 2.0,
            own_thigh_l / 2.0,
            own_thigh_r / 2.0,
            opp_foot_l  / 2.0,
            opp_foot_r  / 2.0,
            opp_shin_l  / 2.0,
            opp_shin_r  / 2.0,
            opp_thigh_l / 2.0,
            opp_thigh_r / 2.0,
            own_fist_r  / 2.0,
            own_fist_l  / 2.0,
            opp_fist_r  / 2.0,
            opp_fist_l  / 2.0,
        ])
        return obs.astype(np.float32)
    def _fist_attack_bonus(self, own_prefix, target_pos):
        """Rakip kafasına doğru en hızlı yumruğun hızına orantılı bonus.

        dot(yumruk_hızı, hedefe_birim_vektör) ∈ [0, 2] m/s aralığı
        0.20 katsayıyla ödüllenir → maks +0.40/adım.
        v3-yeni-mimari: 0.05→0.12 yetersiz kaldı (ep450: sıfır isabet,
        force-curriculum eklenmeden önce). 0.20'ye çıkarıldı.
        """
        # DÖVÜŞ FAZI-3b (2026-07-16, KULLANICI GÖZLEMİ + düzeltme): Ajan YUMRUK
        # değil TEKME/kafa kullanıyor → bonus YUMRUK + AYAK (tekme) kapsar. AMA
        # faz-3'te ajan ayağı KAFAYA doğru (yüksek) savurup HAVAYA tekme atıp
        # bonusu SÖMÜRDÜ (2.89m/s, 0 isabet). DÜZELTME: (1) AYAK hedefi GÖVDE
        # (aşağı, z≈0.5 — alçak tekme daha kolay DEĞER), yumruk hedefi KAFA;
        # (2) dense koef 0.30→0.15 (havaya-savurma sömürüsünü kır, LANDING
        # (dmg×2, çok büyük) baskın olsun).
        foot_target = np.array([target_pos[0], target_pos[1], 0.5], dtype=np.float64)
        best = 0.0
        for limb in ('fist_right', 'fist_left', 'foot_right', 'foot_left'):
            bid = self.key_body_ids.get(f'{own_prefix}_{limb}')
            if bid is None:
                continue
            tgt = foot_target if 'foot' in limb else target_pos  # tekme aşağı, yumruk kafa
            to_target = tgt - self.data.xpos[bid]
            dist = np.linalg.norm(to_target)
            if dist < 1e-6:
                continue
            v_toward = float(np.dot(self.data.cvel[bid][3:6], to_target / dist))
            best = max(best, v_toward)
        return 0.15 * min(max(best, 0.0), 2.0)   # maks +0.30 (anti-sömürü)

    def _get_player_position(self, player_prefix):
        # FIX 5: O(1) — cached torso ID
        bid = self.p1_head_id if player_prefix == 'p1' else self.p2_head_id
        return self.data.xpos[bid].copy() if bid is not None else np.zeros(3)

    def _get_player_velocity(self, player_prefix):
        # FIX 5: O(1) — cached root DOF address
        dof = self.p1_root_dof_adr if player_prefix == 'p1' else self.p2_root_dof_adr
        if dof >= 0 and dof + 3 <= self.model.nv:
            return self.data.qvel[dof:dof+3].copy()
        return np.zeros(3)

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
        return None

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

# ─────────────────────────────────────────────────────────────────────────
# ADAPTİF ENTROPİ KATSAYISI (v2c geliştirme — SAC'taki otomatik-alpha'nın
# PPO'ya taşınması)
# ─────────────────────────────────────────────────────────────────────────
# Tarihçe: entropi krizi projede 4 kez patladı (elle sigma dondurma/eritme
# döngüsü: 1.09→0.83→0.3251→0.573→0.3251...). Sabit ent_coef=0.0 + elle
# actor_log_std.requires_grad_(False) dondurması, entropinin kontrolsüz
# artmasını durdurdu ama σ'yı da tamamen statik bıraktı — ne keşif ne
# inceltme ihtiyacına adapte olabiliyordu.
# Çözüm: SAC'ın otomatik-alpha mekanizması (öğrenilen bir katsayı, hedef
# entropiye göre kendi kendini ayarlar). entropy() burada dist.entropy().mean()
# ile hesaplanıyor (batch VE eylem-boyutu üzerinden ortalama, SAC'taki toplam
# değil) — bu yüzden hedef de PER-DİM analitik Gauss entropisi olarak
# tanımlanıyor: 0.5*log(2πe·σ²).
# TARGET_SIGMA=0.20: daha önce elle denenen 0.18 (fazla kısıtlı) ve 0.30
# (fazla gürültülü, regresyona yol açmıştı) arasında bir orta nokta.
#
# ★ ÇİFT-YÖNLÜ DÜZELTME (2026-07-16): Eski parametrizasyon log_ent_coef idi →
# ent_coef = exp(log_ent_coef) HER ZAMAN POZİTİF → entropi terimi hep BONUS →
# σ'yı yalnızca YUKARI itebiliyordu, AŞAĞI çekemiyordu. σ hedefin üstüne
# çıkınca mekanizma yalnızca bonusu kısabiliyor, aktif ceza veremiyordu →
# ödül-ölçeği sıçraması sonrası politika gradyanı σ'yı kaçırdı (ep7400-7450:
# σ 0.23→0.25 ivmeli, kontrol edilemedi). ÇÖZÜM: doğrudan (exp'siz) ent_coef
# parametresi — NEGATİF olabilir. σ hedefi aşınca ent_coef negatife düşer →
# `-ent_coef*entropy` bir CEZA olur → σ aktif olarak aşağı çekilir. Böylece
# "hedef σ" mekanizması gerçekten İKİ YÖNLÜ olur.
TARGET_SIGMA        = 0.20
TARGET_ENTROPY       = 0.5 * math.log(2 * math.pi * math.e) + math.log(TARGET_SIGMA)
ENT_COEF_INIT        = 0.01     # başlangıç katsayısı küçük (pozitif) — patlama riskini azaltır
ENT_COEF_LR          = 3e-4
# Doğrudan ent_coef sınırları: NEGATİF taraf (ceza, σ'yı aşağı çeker) MODEST
# tutulur (-0.2) — aşırı negatif σ'yı sıfıra çökertip "ölü politika" yapabilir
# (eski dead-clamp riski). Pozitif taraf (bonus, keşif) geniş (5.0).
ENT_COEF_MIN_MAX     = (-0.2, 5.0)

# İKİZ-CRİTİC SICAK-BAŞLATMA süresi: eski-format (tek-critic) checkpoint'ten
# devam edilince twin_critic FRESH başlar. İlk bu kadar güncelleme boyunca
# actor DONDURULUR (train_actor=False) — yeni critic sabit-iyi politikaya
# kalibre olur, kötü değer tahminleri iyi aktörü bozmaz. Her güncelleme ~36
# gradyan adımı (4 epoch × ~9 mini-batch) → 15 güncelleme ~540 adım, critic'in
# fit olması için bolca yeterli; save_freq=50'nin altında (ilk checkpoint'ten
# önce biter).
CRITIC_WARMUP_UPDATES = 15

# ─────────────────────────────────────────────────────────────────────────
# İKİZ-CRİTİC (Twin Critic — SAC/TD3'ün clipped-double-Q'sunun PPO'ya
# taşınması, v2c critic-patlaması krizinin KÖK NEDEN düzeltmesi)
# ─────────────────────────────────────────────────────────────────────────
# Kriz tanısı (2026-07-14/15, iki kez: ep3390 ve ep6950-7000): locomotion
# episode'ları giderek daha çok "truncated" (600 adıma hayatta kalarak)
# bitiyor — bu durumda GAE bootstrap değeri GERÇEK ödül değil, CRİTİC'İN
# KENDİ TAHMİNİ. Politika her yeni performans zirvesine ulaştığında
# (az-görülmüş, "uzun hayatta kalma" durumları), critic o bölgede henüz
# kalibre değil — hatalı/iyimser bir V(s) tahmini, advantage'ı bozup PPO'yu
# yanlış yöne sürüklüyor (gradyan-kırpma adım BÜYÜKLÜĞÜNÜ sınırlar ama
# YÖNÜNÜ düzeltmez).
# Çözüm: TEK critic yerine İKİ TAMAMEN BAĞIMSIZ critic gövdesi (paylaşılan
# aktör temsiliyle SIFIR parametre paylaşımı — hataların decorrelated
# olması şart). Bootstrap/rollout değeri olarak min(V1,V2) kullanılır —
# TD3/SAC'in "clipped double estimator"ı: iki bağımsız tahminin minimumu,
# sistematik iyimserliği cezalandırır, tek-critic'in aşırı-güvenini önler.
class TwinCriticNetwork(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        def make_body():
            return nn.Sequential(
                nn.Linear(obs_dim, 128), nn.LayerNorm(128), nn.SiLU(),
                nn.Linear(128, 128),     nn.LayerNorm(128), nn.SiLU(),
                nn.Linear(128, 64),      nn.SiLU(),
                nn.Linear(64, 1)
            )
        self.critic1 = make_body()
        self.critic2 = make_body()
        self.apply(self._init_weights)
        # Son katmanlar küçük init (deger tahmini asiri buyuk baslamasin)
        nn.init.orthogonal_(self.critic1[-1].weight, gain=1.0); nn.init.zeros_(self.critic1[-1].bias)
        nn.init.orthogonal_(self.critic2[-1].weight, gain=1.0); nn.init.zeros_(self.critic2[-1].bias)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        v1 = self.critic1(x).squeeze(-1)
        v2 = self.critic2(x).squeeze(-1)
        return v1, v2

# ============================================
# NETWORK
# ============================================
class CUDAPPONetwork(nn.Module):
    VARSAYILAN_GIZLI = (320, 224, 160, 112)

    def __init__(self, obs_dim, action_dim, gizli=None):
        super(CUDAPPONetwork, self).__init__()
        # gizli (2026-10-17, kullanıcı onaylı plan: tek ağ büyütme): (h1, h2, h3, latent) actor gövdesi. Varsayılan =
        # eski sabit yapı → eski kayıtlar/çağrılar değişmez. save() kayda yazar, load() farklıysa ağı yeniden kurar.
        self.gizli = tuple(int(x) for x in (gizli or self.VARSAYILAN_GIZLI))
        hidden_1, hidden_2, hidden_3, latent_dim = self.gizli
        self.obs_dim, self.action_dim = obs_dim, action_dim

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_1),
            nn.LayerNorm(hidden_1),
            nn.SiLU(),
            nn.Linear(hidden_1, hidden_2),
            nn.LayerNorm(hidden_2),
            nn.SiLU(),
            nn.Linear(hidden_2, hidden_3),
            nn.LayerNorm(hidden_3),
            nn.SiLU(),
            nn.Linear(hidden_3, latent_dim),
            nn.SiLU()
        )

        # Actor Son Katman
        self.actor_last = nn.Linear(64, action_dim) # Kolay erişim için isimlendirdik

        self.actor_mean = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.SiLU(),
            self.actor_last,
            nn.Tanh()
        )

        # Ham deger 1.09 -> (yumusak tanh sinirlamayla) log_std=-0.51,
        # std ≈ 0.60 — clamp() donemindeki "-0.5 -> std≈0.61" başlangıcıyla
        # tutarlı olsun diye tanh formülüne göre yeniden hesaplandı.
        self.actor_log_std = nn.Parameter(torch.full((action_dim,), 1.09))
        # Tavan 1.5 → 0.0: ilk koşuda entropi bonusu σ'yı 1.28'e şişirdi
        # (aksiyonlar rastgeleye döndü). σ ≤ 1.0 sınırı bunu engeller.
        self.log_std_min, self.log_std_max = -5.0, 0.0

        # İKİZ-CRİTİC: actor'in shared temsiliyle SIFIR parametre paylaşımı —
        # RAW gözlemi (obs_dim) doğrudan alır, kendi bağımsız gövdesiyle
        # işler. Bkz. modül üstü TwinCriticNetwork yorumu.
        self.twin_critic = TwinCriticNetwork(obs_dim)

        self.apply(self._init_weights)

        # DİKKAT: self.apply() TÜM Linear katmanları (twin_critic'in ALT
        # modülleri DAHİL — recursive) orthogonal(gain=√2) ile yeniden
        # başlatır — son katmanların küçük init'i bu yüzden apply'dan SONRA
        # tekrar yapılmalı (twin_critic.__init__'teki kendi küçük-init'i bu
        # üst-seviye apply tarafından eziliyor, tıpkı actor_last gibi).
        nn.init.orthogonal_(self.actor_last.weight, gain=0.01)
        nn.init.zeros_(self.actor_last.bias)
        nn.init.orthogonal_(self.twin_critic.critic1[-1].weight, gain=1.0)
        nn.init.zeros_(self.twin_critic.critic1[-1].bias)
        nn.init.orthogonal_(self.twin_critic.critic2[-1].weight, gain=1.0)
        nn.init.zeros_(self.twin_critic.critic2[-1].bias)

        self.to(DEVICE)

    def scale_weights(self, factor=0.001):
        """
        Manuel müdahale fonksiyonu:
        Ağırlıkları ve biasları belirli bir oranda küçültür.
        Örn: factor=0.1 yaparsan değerler %90 küçülür.
        """
        with torch.no_grad():
            # Actor ağırlıklarını küçült
            self.actor_last.weight.data *= factor
            self.actor_last.bias.data *= factor

            # Critic ağırlıklarını küçült (ikiz-critic — ikisi de)
            self.twin_critic.critic1[-1].weight.data *= factor
            self.twin_critic.critic1[-1].bias.data *= factor
            self.twin_critic.critic2[-1].weight.data *= factor
            self.twin_critic.critic2[-1].bias.data *= factor
        print(f"--- Model ağırlıkları {factor} katsayısı ile manuel olarak güncellendi ---")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        shared = self.shared(x)
        action_mean = self.actor_mean(shared)
        # YUMUŞAK sınırlama (tanh) — sert clamp() kullanılıyordu ama bir
        # parametre sınırı bir kez aşınca clamp'in o bölgedeki gradyanı TAM
        # SIFIR olduğu için actor_log_std bir daha asla geri inemiyordu
        # ("dead clamp" — v2b ep650→ep750'de isabetin sıfıra çökmesinin kök
        # nedeni buydu: σ kalıcı olarak tavanda kilitlenmişti). tanh'ın
        # türevi hiçbir yerde tam sıfır olmadığından gradyan akışı sürer.
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (
            torch.tanh(self.actor_log_std) + 1.0)
        action_std = torch.exp(log_std).expand_as(action_mean)
        # İKİZ-CRİTİC: RAW gözlemi (x) doğrudan alır, her ikisi de (N,) döner.
        v1, v2 = self.twin_critic(x)
        return action_mean, action_std, v1, v2

    def eylem(self, x):
        """forward() ile AYNI (ortalama, std), ikiz critic HESAPLANMADAN — rollout/rakip için (2026-10-21 hız)."""
        action_mean = self.actor_mean(self.shared(x))
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (
            torch.tanh(self.actor_log_std) + 1.0)
        return action_mean, torch.exp(log_std).expand_as(action_mean)
# ============================================
# AGENT (ÇOK ÖNEMLİ - TAM DÜZELTİLMİŞ)
# ============================================
class CUDAPPOAgent:
    def __init__(self, obs_dim, action_dim, lr=5e-4, name="Agent", gizli=None):
        self.network = CUDAPPONetwork(obs_dim, action_dim, gizli=gizli)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self._lr = lr
        self.action_dim = action_dim
        self.name = name

        # ✅ GAE parametreleri
        self.gamma = 0.99
        self.gae_lambda = 0.95

        # ✅ Observation normalization
        self.obs_rms = RunningMeanStd(obs_dim)

        # ✅ DEĞER (RETURN) NORMALİZASYONU (2026-07-16, kredi-atama kök-fix):
        # Critic hedefleri (returns) koşan mean/std ile normalize edilir →
        # critic HER ZAMAN ~birim-ölçek hedef görür, ödül büyüklüğünden BAĞIMSIZ.
        # Böylece değer öğrenme stabil/doğru olur, advantage'lar güvenilir
        # kalır (peak-then-decline sürüklenmesinin kök nedeni: gürültülü/
        # kalibre olmayan critic → hatalı advantage → politika ödül-tepesinden
        # kayıyordu). Critic NORMALİZE değer tahmin eder; GAE için get_value/
        # get_actions_batch bu tahmini DENORM eder (raw ödül-ölçeğinde bootstrap).
        self.ret_rms = RunningMeanStd((1,))

        # ✅ Adaptif entropi katsayısı (SAC-tarzı otomatik-alpha, ÇİFT-YÖNLÜ —
        # bkz. yukarıdaki modül-seviyesi yorum). ent_coef DOĞRUDAN öğrenilir
        # (exp YOK → NEGATİF olabilir → σ hedefi aşınca ceza). Ayrı, küçük bir
        # optimizer ile öğrenilir — ana actor/critic optimizer'ından bağımsız.
        self.target_entropy = TARGET_ENTROPY
        self.ent_coef_param = torch.tensor(float(ENT_COEF_INIT),
                                           requires_grad=True, device=DEVICE)
        self.ent_optimizer = optim.Adam([self.ent_coef_param], lr=ENT_COEF_LR)

        # İKİZ-CRİTİC sıcak-başlatma bayrağı: load() eski-format (tek-critic)
        # checkpoint yüklerse twin_critic FRESH başlar; True olur ve
        # train_improved ilk N güncellemede actor'ı dondurur (critic kalibre
        # olsun). Yeni-format yüklemede veya sıfırdan başlangıçta False kalır.
        self.needs_critic_warmup = False

        # ✅ Mixed precision
        self.scaler = amp.GradScaler("cuda") if USE_AMP else None

        # ✅ LR Scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=300,
            gamma=0.97
        )

        print(f"✓ {name} - Device: {DEVICE}, AMP: {USE_AMP}")

    def _ret_mean_std(self):
        """DEĞER-NORM: koşan return mean/std (python float). Critic çıktısını
        raw ölçeğe çevirmek (denorm) ve return hedefini normalize etmek için."""
        m = float(self.ret_rms.mean[0])
        s = float(np.sqrt(self.ret_rms.var[0]) + 1e-8)
        return m, s

    def get_action(self, state, training=True):
        """✅ Observation normalization eklendi"""
        # ✅ State'i normalize et
        state_normalized = self.obs_rms.normalize(state)
        state_tensor = torch.FloatTensor(state_normalized).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            action_mean, action_std, _v1, _v2 = self.network(state_tensor)

        if training:
            # KRİTİK (2026-10): log_prob KIRPILMIŞ aksiyondan hesaplanmalı —
            # update() da saklanan (kırpılmış) aksiyonun log_prob'unu hesaplıyor.
            # Eski sıra (önce log_prob, sonra clamp) gradyan-ÖNCESİ KL'i ~1.1-1.7
            # yapıyordu (target_kl=0.03) → ilk mini-batch'te KL-stop → actor/critic
            # HİÇ eğitilmiyordu (getup_solo ep25→eppar3875 ağırlıkları bit-bit aynı).
            # get_actions_batch ile aynı sıra.
            dist = torch.distributions.Normal(action_mean, action_std)
            action = torch.clamp(dist.sample(), -1, 1)
            log_prob = dist.log_prob(action).sum(dim=-1)
            return action.cpu().numpy()[0], log_prob.cpu().item()
        else:
            return action_mean.cpu().numpy()[0], 0.0

    def get_value(self, state):
        """✅ Value estimation için yeni fonksiyon"""
        state_normalized = self.obs_rms.normalize(state)
        state_tensor = torch.FloatTensor(state_normalized).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            _, _, v1, v2 = self.network(state_tensor)
            # İKİZ-CRİTİC bootstrap: min(V1,V2) — clipped double estimator
            # (iyimser tahmini cezalandır, critic aşırı-güvenini önle).
            value = torch.min(v1, v2)

        # DEĞER-NORM: critic NORMALİZE değer tahmin eder → GAE için raw ödül-
        # ölçeğine DENORM et. (min monoton olduğu için min sonrası denorm = ayrı
        # denorm sonrası min; ret_std>0.)
        m, s = self._ret_mean_std()
        return value.cpu().item() * s + m

    def compute_gae(self, rewards, values, next_value, dones):
        """✅ GAE hesaplama - DÜZELTME: dones parametresi eklendi"""
        advantages = []
        gae = 0

        # values listesine next_value'yu ekle
        values = values + [next_value]

        for step in reversed(range(len(rewards))):
            # done varsa next_value = 0
            mask = 1 - dones[step]
            delta = rewards[step] + self.gamma * values[step + 1] * mask - values[step]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages.insert(0, gae)

        return advantages

    def update(self, states, actions, old_log_probs, returns, advantages, old_values,
              batch_size=1024, epochs=4, train_actor=True, target_kl=0.03):
        """PPO update: value clipping + KL early stopping + ADAPTİF entropi katsayısı
        (SAC-tarzı otomatik-alpha — bkz. modül üstü yorum). ent_coef artık sabit
        bir parametre değil, self.ent_coef_param üzerinden öğrenilir.

        train_actor=False: SADECE ikiz-critic eğitilir (actor/entropi katkısı ve
        KL erken-durdurma devre dışı). İKİZ-CRİTİC SICAK-BAŞLATMA için: taze critic
        yüklendiğinde birkaç güncelleme boyunca actor DONDURULUR, böylece yeni critic
        sabit-iyi politikaya kalibre olur ve kötü değer tahminleri iyi aktörü
        bozmaz (fresh-critic geçiş riskinin kök çözümü)."""

        # ✅ Observation RMS güncelle
        states_arr = np.array(states)
        if not getattr(self, 'obs_rms_sabit', False):   # 2026-10-18: damıtılmış ağın ince ayarında normalizasyon donuk
            self.obs_rms.update(states_arr)

        # ✅ FIX: Tek tek normalize yerine tüm batch'i vektörel olarak normalize et (çok daha hızlı)
        states_normalized = self.obs_rms.normalize(states_arr)  # (N, obs_dim)
        # SIMETRI KAYBI (2026-10-14, getup): self.sym = (obs_perm, obs_sign, act_perm, act_sign, w)
        # ayarliysa L_sym = w * mean((mu(s) - M_a(mu(M_s(s))))^2) aktor kaybina eklenir (Yu ve ark.
        # 2018). Ayarli degilse (dovus) davranis AYNEN eskisi.
        sym = getattr(self, 'sym', None)
        if sym is not None:
            states_mir = torch.FloatTensor(self.obs_rms.normalize(states_arr[:, sym[0]] * sym[1])).to(DEVICE)
            sym_ap = torch.as_tensor(sym[2], device=DEVICE); sym_as = torch.as_tensor(sym[3], device=DEVICE)
        sym_sum, sym_n = 0.0, 0
        # BC EK KAYBI (2026-10-18, tek ağ ince ayarı): self.bc = dict(O, L (GPU tensör), w, n) ayarlıysa öğretmen
        # etiketlerine (damitma_veri) MSE eklenir → PPO ince ayarında kalkış/yürüme unutulmaz. Ayarlı değilse aynen eskisi.
        bc = getattr(self, 'bc', None)
        bc_sum, bc_n = 0.0, 0

        # DEĞER-NORM: ret_rms'i RAW returns ile güncelle → koşan return ölçeği.
        # Sonra m,s (mean,std) ile returns/old_vals NORMALİZE edilir (critic loss
        # birim-ölçek). obs_rms ile aynı desen: önce update, sonra normalize.
        returns_raw = np.array(returns, dtype=np.float32)
        self.ret_rms.update(returns_raw.reshape(-1, 1))
        ret_m, ret_s = self._ret_mean_std()

        # Tensor'lara çevir
        states = torch.FloatTensor(states_normalized).to(DEVICE)
        actions = torch.FloatTensor(np.array(actions)).to(DEVICE)
        old_log_probs = torch.FloatTensor(np.array(old_log_probs)).to(DEVICE)
        # returns/old_vals RAW ölçekte gelir (get_value denorm etti). Critic
        # NORMALİZE tahmin ettiği için burada hedefi de normalize ediyoruz.
        returns = ((torch.FloatTensor(returns_raw).to(DEVICE)) - ret_m) / ret_s
        advantages  = torch.FloatTensor(np.array(advantages)).to(DEVICE)
        old_vals    = ((torch.FloatTensor(np.array(old_values, dtype=np.float32)).to(DEVICE)) - ret_m) / ret_s

        dataset_size = len(states)

        # ✅ Epoch sayısı azaltıldı (10 → 4)
        # FIX: mini-batch başına KL tavanı. Eski kod KL'i sadece EPOCH sonunda
        # kontrol ediyordu → tek epoch (~16 mini-batch adımı) üst üste birikip
        # KL'i 4'e (!) patlatıp iyi politikayı yok ediyordu — ep1200→1300
        # regresyonunun KÖK NEDENİ buydu (sigma/rakipten bağımsız). Artık her
        # mini-batch öncesi birikmiş KL kontrol edilir; target_kl aşılırsa
        # güncelleme YAPILMADAN durulur → felaket güncelleme imkânsız.
        # target_kl artik parametre (varsayilan 0.03 = eski davranis). Gauss
        # politikada KL ~ (dmu/sigma)^2: sigma kucuk egitimler (getup 0.08) ayni
        # adimda ~6x KL uretir -> olcekli target_kl gecmeli.
        kl_stop   = False
        n_steps   = 0      # tamamlanan mini-batch (optimizer) adimi — KL-stop gorunurlugu
        last_kl   = 0.0
        for epoch in range(epochs):
            if kl_stop:
                break
            indices = torch.randperm(dataset_size)

            for start_idx in range(0, dataset_size, batch_size):
                end_idx = min(start_idx + batch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]

                clip_param       = 0.2
                batch_old_vals   = old_vals[batch_indices]

                # ÇİFT-YÖNLÜ: doğrudan ent_coef (NEGATİF olabilir → σ hedefi
                # aşınca `-ent_coef*entropy` bir cezaya döner, σ'yı aşağı çeker).
                ent_coef = self.ent_coef_param.detach()

                if USE_AMP:
                    with amp.autocast("cuda"):
                        action_mean, action_std, v1_new, v2_new = self.network(batch_states)
                        dist          = torch.distributions.Normal(action_mean, action_std)
                        new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                        entropy       = dist.entropy().mean()

                        # Actor loss (PPO clip)
                        ratio  = torch.exp(new_log_probs - batch_old_log_probs)
                        surr1  = ratio * batch_advantages
                        surr2  = torch.clamp(ratio, 1 - clip_param, 1 + clip_param) * batch_advantages
                        actor_loss = -torch.min(surr1, surr2).mean()

                        # İKİZ-CRİTİC loss — HER İKİ head de returns'e eğitilir,
                        # value-clipping her birine ayrı uygulanır. v1/v2 zaten
                        # (N,) — squeeze GEREKMİYOR. critic_loss = cl1+cl2 (iki
                        # head toplamı → raporlanan değer ESKİ TEK-critic ölçeğin
                        # ~2 KATI; izleme eşiği buna göre ~2x kaydırılmalı).
                        # Loss'ta 0.5 çarpanı → her head efektif 0.5 katsayı alır
                        # (eski tek-critic ile aynı gradyan gücü).
                        v1_clip = batch_old_vals + (v1_new - batch_old_vals).clamp(-clip_param, clip_param)
                        v2_clip = batch_old_vals + (v2_new - batch_old_vals).clamp(-clip_param, clip_param)
                        cl1 = torch.max((batch_returns - v1_new)**2, (batch_returns - v1_clip)**2).mean()
                        cl2 = torch.max((batch_returns - v2_new)**2, (batch_returns - v2_clip)**2).mean()
                        critic_loss = cl1 + cl2

                        # SICAK-BAŞLATMA: train_actor=False iken SADECE critic
                        # eğitilir (actor/entropi katkısı yok).
                        if train_actor:
                            loss = actor_loss + 0.5 * critic_loss - ent_coef * entropy
                            if sym is not None:
                                mu_m = self.network(states_mir[batch_indices])[0]
                                sym_loss = ((action_mean - mu_m[:, sym_ap] * sym_as) ** 2).mean()
                                loss = loss + sym[4] * sym_loss
                                sym_sum += float(sym_loss.detach()); sym_n += 1
                            if bc is not None:
                                bc_loss = self._bc_kaybi(bc); loss = loss + bc["w"] * bc_loss
                                bc_sum += float(bc_loss.detach()); bc_n += 1
                        else:
                            loss = 0.5 * critic_loss

                    # Mini-batch KL kontrolü — bu adım birikmiş KL'i target'ı
                    # aşacaksa güncellemeyi YAPMA ve dur (felaket güncellemeyi önler).
                    # SICAK-BAŞLATMA'da actor sabit → KL sürüklenmesi yok, atla.
                    with torch.no_grad():
                        last_kl = (batch_old_log_probs - new_log_probs).mean().abs().item()
                    if train_actor and last_kl > target_kl:
                        kl_stop = True
                        break

                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                    self.scaler.step(self.optimizer)
                    n_steps += 1
                    self.scaler.update()

                    # ÇİFT-YÖNLÜ adaptif entropi katsayısı güncellemesi: entropy
                    # hedefin ALTINDAYSA ent_coef büyür (daha çok keşif bonusu),
                    # ÜSTÜNDEYSE küçülür ve NEGATİFE geçebilir (aktif ceza → σ'yı
                    # aşağı çeker). entropy.detach() — bu adım actor/critic
                    # gradyanına karışmaz, ayrı optimizer.
                    # ent_coef_loss = ent_coef * (entropy - target): gradyan
                    # inişi → ent_coef -= lr*(entropy-target). entropy>target
                    # (σ fazla) → ent_coef DÜŞER (negatife dek). (exp YOK →
                    # doğrudan parametre, negatif olabilir. Eski log_ent_coef
                    # sadece pozitif kalabiliyordu = tek-yönlü.)
                    # SICAK-BAŞLATMA'da actor sabit → entropi hedefi ayarı anlamsız,
                    # atla.
                    if train_actor:
                        ent_coef_loss = (self.ent_coef_param *
                                         (entropy.detach() - self.target_entropy))
                        self.ent_optimizer.zero_grad()
                        ent_coef_loss.backward()
                        self.ent_optimizer.step()
                        with torch.no_grad():
                            self.ent_coef_param.clamp_(*ENT_COEF_MIN_MAX)
                else:
                    action_mean, action_std, v1_new, v2_new = self.network(batch_states)
                    dist          = torch.distributions.Normal(action_mean, action_std)
                    new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                    entropy       = dist.entropy().mean()

                    ratio  = torch.exp(new_log_probs - batch_old_log_probs)
                    surr1  = ratio * batch_advantages
                    surr2  = torch.clamp(ratio, 1 - clip_param, 1 + clip_param) * batch_advantages
                    actor_loss = -torch.min(surr1, surr2).mean()

                    # İKİZ-CRİTİC loss (bkz. AMP branch yorumu)
                    v1_clip = batch_old_vals + (v1_new - batch_old_vals).clamp(-clip_param, clip_param)
                    v2_clip = batch_old_vals + (v2_new - batch_old_vals).clamp(-clip_param, clip_param)
                    cl1 = torch.max((batch_returns - v1_new)**2, (batch_returns - v1_clip)**2).mean()
                    cl2 = torch.max((batch_returns - v2_new)**2, (batch_returns - v2_clip)**2).mean()
                    critic_loss = cl1 + cl2

                    # SICAK-BAŞLATMA: bkz. AMP branch yorumu
                    if train_actor:
                        loss = actor_loss + 0.5 * critic_loss - ent_coef * entropy
                        if sym is not None:
                            mu_m = self.network(states_mir[batch_indices])[0]
                            sym_loss = ((action_mean - mu_m[:, sym_ap] * sym_as) ** 2).mean()
                            loss = loss + sym[4] * sym_loss
                            sym_sum += float(sym_loss.detach()); sym_n += 1
                        if bc is not None:
                            bc_loss = self._bc_kaybi(bc); loss = loss + bc["w"] * bc_loss
                            bc_sum += float(bc_loss.detach()); bc_n += 1
                    else:
                        loss = 0.5 * critic_loss

                    with torch.no_grad():
                        last_kl = (batch_old_log_probs - new_log_probs).mean().abs().item()
                    if train_actor and last_kl > target_kl:
                        kl_stop = True
                        break

                    self.optimizer.zero_grad()
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                    self.optimizer.step()
                    n_steps += 1

                    # ÇİFT-YÖNLÜ (bkz. AMP branch yorumu): ent_coef negatife
                    # geçebilir → σ hedefi aşınca aktif ceza.
                    if train_actor:
                        ent_coef_loss = (self.ent_coef_param *
                                         (entropy.detach() - self.target_entropy))
                        self.ent_optimizer.zero_grad()
                        ent_coef_loss.backward()
                        self.ent_optimizer.step()
                        with torch.no_grad():
                            self.ent_coef_param.clamp_(*ENT_COEF_MIN_MAX)

            # KL erken-durdurma artık mini-batch içinde (yukarıda). Burada
            # sadece raporlama için son mini-batch KL'ini sakla.
            kl_approx = torch.tensor(last_kl)

        _lc = locals()
        return {
            'actor_loss':  _lc['actor_loss'].item()  if 'actor_loss'  in _lc else 0.0,
            'sym_loss':    (sym_sum / sym_n) if sym_n else 0.0,
            'bc_loss':     (bc_sum / bc_n) if bc_n else 0.0,
            'critic_loss': _lc['critic_loss'].item() if 'critic_loss' in _lc else 0.0,
            'entropy':     _lc['entropy'].item()     if 'entropy'     in _lc else 0.0,
            'ent_coef':    float(self.ent_coef_param.item()),   # negatif olabilir
            'grad_norm':   _lc['grad_norm'].item()   if 'grad_norm'   in _lc else 0.0,
            'kl':          _lc['kl_approx'].item()   if 'kl_approx'  in _lc else 0.0,
            'n_steps':     n_steps,   # 0 = hic guncelleme yok (eski donmus-politika bug'i boyle gorunurdu)
        }

    def _bc_kaybi(self, bc):
        """Öğretmen etiketlerinden rastgele mini grup: MSE(aktör ortalaması, etiket); normalizasyon güncel obs_rms ile."""
        n = bc.get("n", 512); dev = bc["O"].device
        i = torch.randint(0, bc["O"].shape[0], (n,), device=dev)
        ki = bc.get("kalkis_idx")
        if ki is not None and len(ki) > 0:                     # mini grubun bir payı kalkış anlarından (unutmayı önler)
            nk = int(n * bc.get("kalkis_pay", 0.5)); i[:nk] = ki[torch.randint(0, len(ki), (nk,), device=dev)]
        m = torch.as_tensor(self.obs_rms.mean, dtype=torch.float32, device=bc["O"].device)
        sd = torch.as_tensor(np.sqrt(self.obs_rms.var) + 1e-8, dtype=torch.float32, device=bc["O"].device)
        mu = self.network((bc["O"][i] - m) / sd)[0]
        k = bc.get("eylem_sayisi")                             # 2026-10-22 govde2: yalnız eski eklemler (yeniler serbest)
        if k:
            # ...AMA kalkış anlarında HEPSİ (ölçüldü: yeni eklemler kalkışta serbest kalınca kalkış 12/12 → 0/12;
            # kalkışta bel/omuz nötr = öğretmen etiketi doğru). Mini grubun ilk nk satırı kalkış anlarıdır.
            nk = int(n * bc.get("kalkis_pay", 0.5)) if (ki is not None and len(ki) > 0) else 0
            fark = (mu - bc["L"][i]) ** 2
            return torch.cat([fark[:nk].reshape(-1), fark[nk:, :k].reshape(-1)]).mean()
        return ((mu - bc["L"][i]) ** 2).mean()

    def save(self, filename):
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'obs_rms_mean': torch.tensor(self.obs_rms.mean),
            'obs_rms_var': torch.tensor(self.obs_rms.var),
            'obs_rms_count': torch.tensor(self.obs_rms.count),
            'ent_coef_param': self.ent_coef_param.detach().cpu(),  # ÇİFT-YÖNLÜ (negatif olabilir)
            # DEĞER-NORM: return normalizasyon istatistikleri
            'ret_rms_mean': torch.tensor(self.ret_rms.mean),
            'ret_rms_var': torch.tensor(self.ret_rms.var),
            'ret_rms_count': torch.tensor(self.ret_rms.count),
            'gizli': list(self.network.gizli),
        }, filename)

    def load(self, filename):
        checkpoint = torch.load(filename, map_location=DEVICE, weights_only=False)
        sd = checkpoint['network_state_dict']
        gz = tuple(checkpoint.get('gizli', CUDAPPONetwork.VARSAYILAN_GIZLI))
        if gz != self.network.gizli:                  # kayıt farklı boyutta → ağı o boyutta yeniden kur
            self.network = CUDAPPONetwork(self.network.obs_dim, self.network.action_dim, gizli=gz)
            self.optimizer = optim.Adam(self.network.parameters(), lr=self._lr)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=300, gamma=0.97)
        has_twin = any(k.startswith('twin_critic') for k in sd)

        if has_twin:
            # YENİ format (ikiz-critic) — tam yükle
            self.network.load_state_dict(sd)
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception as e:
                print(f"⚠ optimizer state yüklenemedi ({e}) — sıfırdan başlıyor")
            self.needs_critic_warmup = False
            # DEĞER-NORM GEÇİŞİ: checkpoint ret_rms İÇERMİYORSA, twin critic
            # RAW getirilere göre eğitilmiş (~±yüzlerce çıktı) → değer-norm'da
            # bu çıktı NORMALİZE sanılır (denorm ×std ile patlar). Çıktı
            # katmanlarını SIFIRLA (→ ~0 çıktı = normalize ölçek), gövdeyi/
            # öznitelikleri KORU; warmup çıktı-ölçeğini yeni (normalize) hedefe
            # yeniden öğrenir. needs_critic_warmup=True (actor donuk, güvenli).
            if 'ret_rms_mean' not in checkpoint:
                with torch.no_grad():
                    for head in (self.network.twin_critic.critic1[-1],
                                 self.network.twin_critic.critic2[-1]):
                        nn.init.orthogonal_(head.weight, gain=0.01); nn.init.zeros_(head.bias)
                self.needs_critic_warmup = True
                print("↻ DEĞER-NORM GEÇİŞİ: twin critic çıktı-katmanları sıfırlandı "
                      "(gövde korundu), warmup ile normalize ölçeğe kalibre olacak.")
        else:
            # ESKİ format (tek-critic): actor+shared AĞIRLIKLARINI TAŞI,
            # twin_critic FRESH kalsın (strict=False). optimizer YÜKLENMEZ
            # (parametre kümesi değişti — eski critic yok, twin_critic var).
            result = self.network.load_state_dict(sd, strict=False)
            fresh = [k for k in result.missing_keys]         # twin_critic.* (beklenen)
            dropped = [k for k in result.unexpected_keys]    # critic.*/critic_last.* (atılan)
            # GÜVENLİK: missing SADECE twin_critic olmalı; actor/shared eksikse hata
            non_twin_missing = [k for k in fresh if not k.startswith('twin_critic')]
            assert not non_twin_missing, f"actor/shared eksik yüklendi: {non_twin_missing}"
            print(f"↻ ESKİ format checkpoint: actor+shared taşındı, İKİZ-CRİTİC SIFIRDAN "
                  f"(fresh: {len(fresh)} anahtar, atılan eski-critic: {len(dropped)}).")
            print(f"  → optimizer SIFIRDAN + SICAK-BAŞLATMA aktif (critic önce kalibre olacak).")
            self.needs_critic_warmup = True

        # ✅ RMS parametrelerini de yükle (numpy array veya tensor olabilir)
        if 'obs_rms_mean' in checkpoint:
            mean_val = checkpoint['obs_rms_mean']
            var_val = checkpoint['obs_rms_var']
            count_val = checkpoint['obs_rms_count']

            # Tensor ise numpy'ye çevir
            if isinstance(mean_val, torch.Tensor):
                self.obs_rms.mean = mean_val.cpu().numpy()
            else:
                self.obs_rms.mean = np.array(mean_val)

            if isinstance(var_val, torch.Tensor):
                self.obs_rms.var = var_val.cpu().numpy()
            else:
                self.obs_rms.var = np.array(var_val)

            if isinstance(count_val, torch.Tensor):
                self.obs_rms.count = count_val.item()
            else:
                self.obs_rms.count = float(count_val)

        # DEĞER-NORM: ret_rms istatistiklerini yükle. GERİYE UYUMLU: eski
        # checkpoint'lerde yok → ret_rms sıfırdan (mean=0,var=1) kalır; ilk
        # update()'te RAW returns'ten öğrenir. force_critic_warmup ile critic
        # zaten yeni (normalize) ölçeğe kalibre olur.
        if 'ret_rms_mean' in checkpoint:
            rm = checkpoint['ret_rms_mean']; rv = checkpoint['ret_rms_var']; rc = checkpoint['ret_rms_count']
            self.ret_rms.mean  = rm.cpu().numpy() if isinstance(rm, torch.Tensor) else np.array(rm)
            self.ret_rms.var   = rv.cpu().numpy() if isinstance(rv, torch.Tensor) else np.array(rv)
            self.ret_rms.count = rc.item() if isinstance(rc, torch.Tensor) else float(rc)

        # ✅ Adaptif entropi katsayısını yükle. GERİYE UYUMLU: yeni format
        # 'ent_coef_param' (doğrudan, çift-yönlü); eski format 'log_ent_coef'
        # (exp ile pozitif) → exp() ile doğrudan değere çevrilir. Hiçbiri yoksa
        # ENT_COEF_INIT'te kalır.
        with torch.no_grad():
            if 'ent_coef_param' in checkpoint:
                self.ent_coef_param.copy_(checkpoint['ent_coef_param'].to(DEVICE))
            elif 'log_ent_coef' in checkpoint:
                # eski log-parametrizasyon → ent_coef = exp(log_ent_coef)
                old = checkpoint['log_ent_coef'].to(DEVICE)
                self.ent_coef_param.copy_(old.exp())

    def get_actions_batch(self, states_list, training=True):
    # Tek obs gelirse listeye çevir
        single = isinstance(states_list, np.ndarray) and states_list.ndim == 1
        if single:
            states_list = [states_list]

    # ✅ FIX: Vektörize normalize — list comprehension kaldırıldı
        states_arr = self.obs_rms.normalize(np.array(states_list))
        state_tensor = torch.FloatTensor(states_arr).to(DEVICE)

        with torch.no_grad():
            action_mean, action_std, v1, v2 = self.network(state_tensor)
            # İKİZ-CRİTİC bootstrap: rollout değeri = min(V1,V2). twin_critic
            # zaten (N,) döndürür — TEKRAR squeeze YAPMA (N==1'de scalar'a
            # düşüp values_np[0] IndexError verir).
            values = torch.min(v1, v2)

        if training:
            dist = torch.distributions.Normal(action_mean, action_std)
            actions = torch.clamp(dist.sample(), -1, 1)
            log_probs = dist.log_prob(actions).sum(dim=-1)
        else:
            actions = action_mean
            log_probs = torch.zeros(len(states_list), device=DEVICE)

        actions_np   = actions.cpu().numpy()        # (N, action_dim)
        log_probs_np = log_probs.cpu().numpy()      # (N,)
        # DEĞER-NORM: critic NORMALİZE değer verir → GAE için raw ödül-ölçeğine
        # DENORM et.
        m, s = self._ret_mean_std()
        values_np    = values.cpu().numpy() * s + m # (N,) raw ölçek — twin zaten squeezed

     # Tek obs geldiyse scalar döndür
        if single:
            return actions_np[0], float(log_probs_np[0]), float(values_np[0])

        return actions_np, log_probs_np, values_np
# ============================================
# HEURISTIC SPARRING PARTNER
# ============================================
# Self-play'de iki öğrenen ajan birbirini kovaladığında (co-adaptation),
# ikisi de en kolay kaçış davranışına kayabiliyor (gözlemlendi: önce
# "heykel dengesi", sonra "kaçış", sonra "çömelerek sahte yaklaşma").
# Sabit, öğrenmeyen bir bot bu döngüyü kırar: kaçmaz, ana ajana garantili
# "isabet nasıl olur" deneyimi sağlar. Mükemmel dövüşçü olması gerekmiyor.
def _actuator_index_map(env):
    """'p1_' önekli aktüatör adı (önek hariç) -> local index. env'e cache'lenir."""
    if not hasattr(env, '_act_idx_cache'):
        idx_map = {}
        i1 = 0
        for i in range(env.model.nu):
            name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name and name.startswith('p1_'):
                idx_map[name[3:]] = i1
                i1 += 1
        env._act_idx_cache = idx_map
    return env._act_idx_cache


def heuristic_sparring_action(env, prefix, t):
    """Sabit kurallı rakip: dengede durmaya çalışır ve kollarını periyodik
    olarak öne savurur. Başlangıç mesafesi (~1.0-1.6m) zaten yumruk
    menzilinde olduğundan yürüme gerekmiyor — koordinasyonsuz bacak
    salınımı denenmişti, P1'i yaklaştırmak yerine düşürüyordu.

    NOT: knee stiffness XML'de çok düşük (=1) — ctrl=0 ile diz yerçekimi
    altında çöküyor (~1.7s). knee+hip_y'ye sabit +0.5 destek dayanıklılığı
    uzatıyor (567→730 adım, ölçüldü) ama tam çözmüyor; MuJoCo'nun kendi
    sayısal driftinden kaynaklanan kalıntı bir sorun olabilir."""
    idx = _actuator_index_map(env)
    a = np.zeros(env.p1_actuator_count, dtype=np.float32)

    a[idx['knee_left']]   = 0.5
    a[idx['knee_right']]  = 0.5
    a[idx['hip_y_left']]  = 0.5
    a[idx['hip_y_right']] = 0.5

    phase = t * 0.15
    swing   = np.sin(phase * 1.7)
    punch_r = max(0.0, swing)
    punch_l = max(0.0, -swing)
    a[idx['shoulder_right']] = -0.85 * punch_r
    a[idx['elbow_right']]    = -0.6  * punch_r
    a[idx['shoulder_left']]  = -0.85 * punch_l
    a[idx['elbow_left']]     = -0.6  * punch_l

    return np.clip(a, -1.0, 1.0)


# ============================================
# PARALLEL ENV (DÜZELTİLDİ)
# ============================================
class ParallelEnvRunner:
    def __init__(self, num_envs, xml_path, difficulty=1.0):
        self.num_envs = num_envs
        self.combat_weight = 1.0
        self.envs = [
            DualHumanoidFightEnv(
                render_mode=None,
                xml_path=xml_path,
                env_id=i,
                difficulty=difficulty
            )
            for i in range(num_envs)
        ]
        print(f"✓ {num_envs} sequential environment oluşturuldu (difficulty: {difficulty:.2f})")

    def reset_all(self):
        results = []
        for env in self.envs:
            env.reset()
            obs_p1, obs_p2 = env.get_obs_both()
            results.append((obs_p1, obs_p2))
        return results

    def step_all(self, actions_list):
        """Tüm env'leri sırayla adım attır, step() 5-tuple döndürür."""
        return [env.step(actions) for env, actions in zip(self.envs, actions_list)]

    def close_all(self):
        for env in self.envs:
            env.close()

# ============================================
# TRAINING (TAM DÜZELTİLMİŞ - TÜM İYİLEŞTİRMELER)
# ============================================
def train_improved(episodes=5000, save_freq=50, xml_path="dual_humanoid_fight.xml",
                   resume_from=None, force_critic_warmup=False,
                   ckpt_prefix="main_agent_v2c_ep", resume_ckpt=None):
    # force_critic_warmup: YENİ-format (ikiz-critic) checkpoint'ten devam
    # edilirken bile critic SICAK-BAŞLATMASINI zorla. Ödül yapısı değiştiğinde
    # (ör. 2026-07-16 ileri-ödül/geriye-ceza düzeltmesi) gereklidir — yüklenen
    # critic ESKİ ödül ölçeğine kalibre; actor eğitilmeden önce critic'in YENİ
    # ödüle yeniden oturması, kalibrasyon-bozukluğunun aktörü sarsmasını önler.
    #
    # ckpt_prefix: kaydedilen checkpoint önekι. DÖVÜŞ fazı ayrı bir önek
    # ("main_agent_v2f_ep") kullanır → YÜRÜME (v2c) checkpoint'lerinin üzerine
    # YAZMAZ, karışmaz (kullanıcı isteği: ep3200 yürüyücü korunsun).
    # resume_ckpt: yüklenecek DOSYA adı (varsayılan pattern'i ezer). Dövüş
    # yürüyücü tabandan (EN_IYI_YURUYUCU_ep3200.pth) başlayıp v2f önekiyle
    # kaydeder — kaynak ve hedef ayrışır.
    best_ckpt_name = ckpt_prefix.replace("_ep", "_best.pth") if "_ep" in ckpt_prefix else "best_main_agent_v2c.pth"
    final_ckpt_name = ckpt_prefix.replace("_ep", "_final.pth") if "_ep" in ckpt_prefix else "main_agent_v2c_final.pth"

    print("\n" + "="*70)
    print("🚀 FULLY IMPROVED DUAL HUMANOID FIGHT TRAINING")
    print("="*70)
    print(f"Device: {DEVICE}")
    print(f"Mixed Precision: {USE_AMP}")
    print(f"Parallel Envs: {NUM_PARALLEL_ENVS}")
    print("="*70 + "\n")

    writer = SummaryWriter('runs/dual_fight_selfplay')
    env = DualHumanoidFightEnv(render_mode=None, xml_path=xml_path)
    obs_dim    = env.observation_space.shape[0]
    action_dim = env.p1_actuator_count          # P1 == P2 (simetrik)
    main_agent = CUDAPPOAgent(obs_dim=obs_dim, action_dim=action_dim,
                              name="Main_Fighter")
    env.close()

    # v2c/geliştirme: actor_log_std artık DONDURULMUYOR. Başlangıç σ≈0.20'ye
    # (TARGET_SIGMA) ayarlanır ama gradyan akışı AÇIK — adaptif entropi
    # katsayısı (self.ent_coef_param, __init__'te kurulu) σ'yı hedef entropiye
    # göre kendi kendine ayarlar. Bu, elle dondurma/eritme döngüsünün
    # (1.09→0.83→0.3251→0.573→...) kalıcı çözümü — bkz. modül üstü yorum.
    with torch.no_grad():
        raw = math.atanh(2 * (math.log(TARGET_SIGMA) -
                              main_agent.network.log_std_min) /
                         (main_agent.network.log_std_max -
                          main_agent.network.log_std_min) - 1)
        main_agent.network.actor_log_std.fill_(raw)   # σ≈TARGET_SIGMA (0.20)

    # ── Opponent pool yardımcıları ────────────────────────────────
    opponent_pool = deque(maxlen=10)

    def _snapshot(agent, ep):
        return {
            'network':       {k: v.detach().cpu().clone()
                              for k, v in agent.network.state_dict().items()},
            'obs_rms_mean':  agent.obs_rms.mean.copy(),
            'obs_rms_var':   agent.obs_rms.var.copy(),
            'obs_rms_count': agent.obs_rms.count,
            'episode':       ep,
        }

    def _make_frozen(snap):
        opp = CUDAPPOAgent(obs_dim=obs_dim, action_dim=action_dim, name="Frozen")
        opp.network.load_state_dict(snap['network'])
        opp.obs_rms.mean  = snap['obs_rms_mean'].copy()
        opp.obs_rms.var   = snap['obs_rms_var'].copy()
        opp.obs_rms.count = snap['obs_rms_count']
        return opp

    # Bootstrap: başlangıç (rastgele) politikayı pool'a ekle
    opponent_pool.append(_snapshot(main_agent, 0))
    current_opponent = _make_frozen(opponent_pool[-1])

    # ✅ CHECKPOINT YÜKLEME
    start_episode = 0
    if resume_from or resume_ckpt:
        try:
            _load_path = resume_ckpt if resume_ckpt else f"main_agent_v2c_ep{resume_from}.pth"
            main_agent.load(_load_path)
            # actor_log_std'yi sıfırla — eski checkpoint'ler sert clamp()
            # döneminden kalma, sınıra "kilitlenmiş" (dead gradient) bir ham
            # değer taşıyor olabilir. Sadece bu tek vektörü resetlemek,
            # network'ün geri kalanını (isabet becerisini kodlayan shared/
            # actor_mean/critic ağırlıkları) koruyarak keşfi yeniden açar.
            # v2c/geliştirme: ARTIK DONDURULMUYOR — bkz. fresh-start bloğundaki
            # aynı gerekçe. σ, TARGET_SIGMA'ya resetlenir, sonrası adaptif
            # entropi katsayısına (self.ent_coef_param, load() ile checkpoint'ten
            # gelmişse onunla, yoksa ENT_COEF_INIT'ten) bırakılır.
            with torch.no_grad():
                raw = math.atanh(2 * (math.log(TARGET_SIGMA) -
                                      main_agent.network.log_std_min) /
                                 (main_agent.network.log_std_max -
                                  main_agent.network.log_std_min) - 1)
                main_agent.network.actor_log_std.fill_(raw)   # σ≈TARGET_SIGMA (0.20)
            # LR'yi DÜŞÜR (uzun-episode düzeltmesiyle birlikte kararlılık):
            # kısa episode + düşük LR + KL-tavanı = ep1200'ü bozmadan inceltme.
            # StepLR base_lrs'i de düşür ki scheduler.step() yüksek değere geri
            # döndürmesin.
            LOCO_LR = 2.0e-4
            main_agent.scheduler.base_lrs = [LOCO_LR] * len(main_agent.scheduler.base_lrs)
            for pg in main_agent.optimizer.param_groups:
                pg['lr'] = LOCO_LR
            print(f"⚙️ LR düşürüldü → {LOCO_LR}  (kararlı locomotion inceltme)")
            # start_episode: resume_from verilirse oradan (yürüme devamı); sadece
            # resume_ckpt verilirse (DÖVÜŞ geçişi: yürüyücü ağırlık yüklenir ama
            # müfredat SIFIRDAN başlar → dövüşü kolay difficulty'de öğrensin) 0.
            start_episode = resume_from if resume_from else 0
            # Pool'u da yüklenmiş ajan snapshot'ından başlat
            opponent_pool.clear()
            opponent_pool.append(_snapshot(main_agent, start_episode))
            current_opponent = _make_frozen(opponent_pool[-1])
            print(f"✅ Ağırlık yüklendi: {_load_path}  (start_episode={start_episode})")
            if main_agent.needs_critic_warmup:
                print(f"🔥 İKİZ-CRİTİC SICAK-BAŞLATMA: ilk {CRITIC_WARMUP_UPDATES} "
                      f"güncellemede actor DONDURULACAK (yeni critic kalibre olsun).")
        except Exception as e:
            print(f"⚠️ Checkpoint yüklenemedi: {e}")
            print("   → Sıfırdan başlıyor...")

    # reward_normalizer kaldırıldı — advantage normalizasyonu update() içinde

    best_win_rate = 0.0
    patience = 300
    no_improvement_count = 0
    min_episodes_before_stop = 500

    main_wins = 0    # main_agent frozen rakibi yendiğinde
    opp_wins  = 0    # frozen rakip kazandığında
    draws     = 0
    # Kayan pencere: erken durdurma / best model kararı kümülatif orana değil
    # son ~200 frozen maça bakar (kümülatif oran zamanla donuyor, yanıltıcıydı).
    recent_results = deque(maxlen=200)

    parallel_envs = None

    # İKİZ-CRİTİC sıcak-başlatma sayacı: taze critic yüklendiyse VEYA ödül
    # değiştiği için zorlanmışsa CRITIC_WARMUP_UPDATES kadar güncelleme actor'ı
    # dondurur (critic önce yeni ödüle/duruma kalibre olsun).
    critic_warmup_remaining = (CRITIC_WARMUP_UPDATES
                               if (main_agent.needs_critic_warmup or force_critic_warmup)
                               else 0)
    if force_critic_warmup and not main_agent.needs_critic_warmup:
        print(f"🔥 ZORLANMIŞ SICAK-BAŞLATMA (ödül değişti): ilk {CRITIC_WARMUP_UPDATES} "
              f"güncellemede actor DONDURULACAK (critic yeni ödüle kalibre olsun).")

    for episode in range(start_episode, episodes):

        # ✅ CURRICULUM
        if episode % 50 == 0 or parallel_envs is None:
            if parallel_envs:
                parallel_envs.close_all()
            # İlk koşuda müfredat çok hızlıydı: denge oturmadan episode'lar
            # uzayınca ödül sürekli düştü. Yeni rampa 2x daha yavaş.
            if episode < 200:
                difficulty   = 1.0
                max_ts       = 500
            elif episode < 600:
                difficulty   = 1.0 + (episode - 200) / 800                # 1.0→1.5
                max_ts       = 500  + int((episode - 200) / 400 * 1500)   # 500→2000
            else:
                difficulty   = min(1.5 + (episode - 600) / 10000, 2.5)
                max_ts       = min(2000 + int((episode - 600) / 200 * 400), 4000)  # 2000→4000

            # FIX 5: max_timesteps curriculum — tüm env'lere uygula
            # Locomotion: episode'ları KISA tut (LOCOMOTION_MAX_STEPS) — uzun
            # ufuk critic'i bozup regresyona yol açıyordu.
            if COMBINE_PHASE:
                eff_max_ts = COMBINE_MAX_STEPS        # 800: dövüş+düş+kalk+devam
            elif GETUP_PHASE:
                eff_max_ts = GETUP_MAX_STEPS          # sabit süre (düş-kalk döngüsü için)
            elif LOCOMOTION_PHASE:
                eff_max_ts = min(max_ts, LOCOMOTION_MAX_STEPS)
            else:
                eff_max_ts = max_ts
            parallel_envs = ParallelEnvRunner(NUM_PARALLEL_ENVS, xml_path, difficulty)
            combat_weight = 1.0
            for env in parallel_envs.envs:
                env.max_timesteps = eff_max_ts
                env.combat_weight = combat_weight
            print(f"🎓 Ep {episode}: difficulty={difficulty:.2f}  max_steps={eff_max_ts}")

        # ✅ OPPONENT POOL GÜNCELLEMESİ (her 50 episode'da bir)
        # Önceki kod: sadece %20 ihtimalle donduruyordu → yetersizdi.
        # Yeni kod: her 50 episode'da main_agent'ı pool'a ekle,
        # %80 en güncel, %20 rastgele eski versiyon.
        # COMBAT_VS_FROZEN: rakip SABİT kalır (ep3200'e kilitli) — snapshot
        # GÜNCELLENMEZ ki non-stationarity olmasın (fazın tüm amacı bu).
        if episode > 0 and episode % 50 == 0 and not COMBAT_VS_FROZEN:
            opponent_pool.append(_snapshot(main_agent, episode))
            if random.random() < 0.2 and len(opponent_pool) > 1:
                snap = random.choice(list(opponent_pool)[:-1])   # daha eski
            else:
                snap = opponent_pool[-1]                          # en güncel
            current_opponent = _make_frozen(snap)
            print(f"🔄 Opponent güncellendi → ep{snap['episode']} snapshot")

        obs_pairs = parallel_envs.reset_all()
        obs_list_p1 = [pair[0] for pair in obs_pairs]
        obs_list_p2 = [pair[1] for pair in obs_pairs]

        # FIX 2: P1 ve P2 trajektörleri AYRI tutulur — GAE'de karışmasın
        # Her env için son obs ve truncated bayrağını da sakla (bootstrap için)
        all_s1 = [[] for _ in range(NUM_PARALLEL_ENVS)]   # states P1
        all_a1 = [[] for _ in range(NUM_PARALLEL_ENVS)]
        all_lp1= [[] for _ in range(NUM_PARALLEL_ENVS)]
        all_r1 = [[] for _ in range(NUM_PARALLEL_ENVS)]
        all_v1 = [[] for _ in range(NUM_PARALLEL_ENVS)]
        all_d1 = [[] for _ in range(NUM_PARALLEL_ENVS)]

        all_s2 = [[] for _ in range(NUM_PARALLEL_ENVS)]   # states P2
        all_a2 = [[] for _ in range(NUM_PARALLEL_ENVS)]
        all_lp2= [[] for _ in range(NUM_PARALLEL_ENVS)]
        all_r2 = [[] for _ in range(NUM_PARALLEL_ENVS)]
        all_v2 = [[] for _ in range(NUM_PARALLEL_ENVS)]
        all_d2 = [[] for _ in range(NUM_PARALLEL_ENVS)]

        last_obs_p1    = [None]  * NUM_PARALLEL_ENVS  # son gözlem (bootstrap için)
        last_obs_p2    = [None]  * NUM_PARALLEL_ENVS
        truncated_flag = [False] * NUM_PARALLEL_ENVS  # FIX 2: KO mu, süre mi?

        done_flags  = [False] * NUM_PARALLEL_ENVS
        step_counts = [0]    * NUM_PARALLEL_ENVS

        # Heuristic sparring partner: self-play co-adaptation'ın kısır
        # döngülerini (heykel dengesi / kaçış / sahte-yaklaşma) kırmak için
        # ilk ~400 episode'da azalan olasılıkla P2 rolüne sabit bir bot
        # yerleştirilir. Rastgelelik episode başına SABİT olmalı (while
        # döngüsü içinde tekrar örneklenirse adımdan adıma tutarsız olur).
        heuristic_prob = max(0.6 * (1.0 - (episode - start_episode) / 400.0), 0.0)
        # Locomotion: HER episode P2 = SABİT hedef (sıfır aksiyon, eğitilmez).
        # Tek ajan (P1) hareketsiz landmark'a doğru yürümeyi öğrenir → self-play
        # non-stationarity (değişen/hareketli rakip, regresyonun kök nedeni)
        # tamamen kalkar. use_frozen=True yolu P2'yi zaten eğitimden dışlar;
        # aşağıda sadece P2 aksiyonunu sıfırlıyoruz.
        # COMBAT_VS_FROZEN: dövüş fazı-1'de HER episode donmuş rakip (self-play
        # non-stationarity YOK). Aksi halde: locomotion'da hep frozen, dövüş
        # self-play'inde her 5'te 1 frozen.
        use_frozen_this_ep    = (episode % 5 == 0) or LOCOMOTION_PHASE or COMBAT_VS_FROZEN or GETUP_PHASE
        use_heuristic_this_ep = (not use_frozen_this_ep) and (random.random() < heuristic_prob)

        while not all(done_flags):
            active_indices = [i for i in range(NUM_PARALLEL_ENVS) if not done_flags[i]]
            active_obs_p1  = [obs_list_p1[i] for i in active_indices]
            active_obs_p2  = [obs_list_p2[i] for i in active_indices]

            # main_agent hem P1 hem P2 tarafı için aksiyon üretir (simetrik)
            batch_a1, batch_lp1, batch_v1 = main_agent.get_actions_batch(active_obs_p1)
            batch_a2, batch_lp2, batch_v2 = main_agent.get_actions_batch(active_obs_p2)

            # current_opponent aksiyonu sadece env'e uygulamak için —
            # ama bu episode'da P2 tarafı da main_agent tarafından kontrol ediliyor.
            # Rakip baskısı için her 5 episode'dan 1'inde P2 frozen opponent kullanılır.
            if use_frozen_this_ep and (LOCOMOTION_PHASE or GETUP_PHASE):
                batch_a2_env = None   # locomotion/getup: P2 SABİT (sıfır aksiyon, aşağıda)
            elif use_frozen_this_ep:
                batch_a2_env, _, _ = current_opponent.get_actions_batch(
                    active_obs_p2, training=False)
            elif use_heuristic_this_ep:
                batch_a2_env = None   # per-env hesaplanacak (aşağıda)
            else:
                batch_a2_env = batch_a2   # main_agent P2 tarafını da oynar

            actions_list = []
            batch_idx = 0

            for env_idx in range(NUM_PARALLEL_ENVS):
                if done_flags[env_idx]:
                    total_act = (parallel_envs.envs[env_idx].p1_actuator_count +
                                 parallel_envs.envs[env_idx].p2_actuator_count)
                    actions_list.append(np.zeros(total_act))
                    continue

                a1     = batch_a1[batch_idx]
                if use_frozen_this_ep and (LOCOMOTION_PHASE or GETUP_PHASE):
                    # P2 = sabit: sıfır aksiyon (locomotion'da statik landmark;
                    # getup'ta zaten uzakta/alakasız).
                    a2_env = np.zeros(parallel_envs.envs[env_idx].p2_actuator_count)
                elif use_heuristic_this_ep:
                    a2_env = heuristic_sparring_action(
                        parallel_envs.envs[env_idx], 'p2', step_counts[env_idx])
                else:
                    a2_env = batch_a2_env[batch_idx]
                lp1    = float(batch_lp1[batch_idx])
                v1     = float(batch_v1[batch_idx])

                a2     = batch_a2[batch_idx]
                lp2    = float(batch_lp2[batch_idx])
                v2     = float(batch_v2[batch_idx])

                combined_action = np.concatenate([a1, a2_env])
                actions_list.append(combined_action)

                # P1 trajektörü
                all_s1[env_idx].append(obs_list_p1[env_idx])
                all_a1[env_idx].append(a1)
                all_lp1[env_idx].append(lp1)
                all_v1[env_idx].append(v1)

                # P2 trajektörü (frozen/heuristic episode'da on-policy değil —
                # sadece P1 güncellenir; P2'nin aksiyonu ana ajandan gelmiyor)
                if not use_frozen_this_ep and not use_heuristic_this_ep:
                    all_s2[env_idx].append(obs_list_p2[env_idx])
                    all_a2[env_idx].append(a2)
                    all_lp2[env_idx].append(lp2)
                    all_v2[env_idx].append(v2)

                batch_idx += 1

            results = parallel_envs.step_all(actions_list)

            for env_idx, (obs_pair, reward, terminated, truncated, info) in enumerate(results):
                if done_flags[env_idx]:
                    continue

                obs_list_p1[env_idx], obs_list_p2[env_idx] = obs_pair

                r1 = reward['p1']
                r2 = reward['p2']
                # FIX: truncated adımda done=1 yazmak GAE'deki bootstrap'i
                # (mask=0) iptal ediyordu. done sadece gerçek terminal (KO/karar).
                # Truncation'da gelecek değer critic'ten bootstrap edilir.
                is_done = 1 if terminated else 0

                all_r1[env_idx].append(float(np.clip(r1, -150.0, 150.0)) * REWARD_SCALE)
                all_d1[env_idx].append(is_done)

                if not use_frozen_this_ep and not use_heuristic_this_ep:
                    all_r2[env_idx].append(float(np.clip(r2, -150.0, 150.0)) * REWARD_SCALE)
                    all_d2[env_idx].append(is_done)

                # FIX 2: Son obs'u ve truncated bayrağını kaydet (GAE bootstrap)
                last_obs_p1[env_idx]    = obs_list_p1[env_idx]
                last_obs_p2[env_idx]    = obs_list_p2[env_idx]
                truncated_flag[env_idx] = bool(truncated)

                step_counts[env_idx] += 1

                if terminated or truncated:
                    done_flags[env_idx] = True
                    # FIX: Simetrik episode'larda iki tarafı da main_agent oynuyor —
                    # o maçlarda "kazanma" anlamsız. Win-rate sadece frozen rakibe
                    # karşı oynanan maçlardan ölçülür.
                    if use_frozen_this_ep:
                        if info['winner'] and 'Player 1' in info['winner']:
                            main_wins += 1
                            recent_results.append(1)
                        elif info['winner'] and 'Player 2' in info['winner']:
                            opp_wins += 1
                            recent_results.append(0)
                        else:
                            draws += 1

        # FIX 2: Per-env GAE — truncated bootstrap.
        # Entropi katsayısı artık burada SABİT değil — CUDAPPOAgent.update()
        # içinde adaptif olarak (self.ent_coef_param, SAC-tarzı otomatik-alpha)
        # hesaplanıyor. Eski elle-ayarlama tarihçesi (0.05→...→0.0 sabit,
        # 4 entropi krizi) modül üstü yorumda belgelendi.

        states_m  = []; actions_m = []; lp_m   = []
        rets_m    = []; advs_m    = []; vals_m  = []

        for env_idx in range(NUM_PARALLEL_ENVS):
            # ─ P1 trajektörü ─────────────────────────────
            if all_s1[env_idx]:
                # FIX 2: truncated → critic'ten bootstrap; KO → 0
                if truncated_flag[env_idx] and last_obs_p1[env_idx] is not None:
                    nv1 = main_agent.get_value(last_obs_p1[env_idx])
                else:
                    nv1 = 0.0
                adv1 = main_agent.compute_gae(all_r1[env_idx], all_v1[env_idx], nv1, all_d1[env_idx])
                ret1 = [a + v for a, v in zip(adv1, all_v1[env_idx])]
                states_m.extend(all_s1[env_idx]);  actions_m.extend(all_a1[env_idx])
                lp_m.extend(all_lp1[env_idx]);     rets_m.extend(ret1)
                advs_m.extend(adv1);               vals_m.extend(all_v1[env_idx])

            # ─ P2 trajektörü (frozen/heuristic episode'da yoktur) ──
            if not use_frozen_this_ep and not use_heuristic_this_ep and all_s2[env_idx]:
                if truncated_flag[env_idx] and last_obs_p2[env_idx] is not None:
                    nv2 = main_agent.get_value(last_obs_p2[env_idx])
                else:
                    nv2 = 0.0
                adv2 = main_agent.compute_gae(all_r2[env_idx], all_v2[env_idx], nv2, all_d2[env_idx])
                ret2 = [a + v for a, v in zip(adv2, all_v2[env_idx])]
                states_m.extend(all_s2[env_idx]);  actions_m.extend(all_a2[env_idx])
                lp_m.extend(all_lp2[env_idx]);     rets_m.extend(ret2)
                advs_m.extend(adv2);               vals_m.extend(all_v2[env_idx])

        # Tüm env ödüllerini flatten et (logging için)
        rewards_m = [r for env_r in all_r1 for r in env_r]
        if not use_frozen_this_ep and not use_heuristic_this_ep:
            rewards_m += [r for env_r in all_r2 for r in env_r]

        metrics = {}
        if len(states_m) > 0:
            advs_arr = np.array(advs_m)
            if len(advs_arr) > 1:
                advs_arr = (advs_arr - advs_arr.mean()) / (advs_arr.std() + 1e-8)
            # FIX: Return'ler artık normalize EDİLMİYOR. Her batch'te ayrı
            # normalize etmek critic hedefinin ölçeğini sürekli kaydırıyordu
            # ve value clipping'deki old_values ham ölçekte kaldığı için
            # loss tutarsızdı. Ölçek REWARD_SCALE ile toplarken küçültülüyor.
            rets_arr = np.array(rets_m)
            # SICAK-BAŞLATMA: taze twin_critic yüklendiyse ilk N güncellemede
            # actor DONDURULUR (train_actor=False) — critic önce kalibre olsun.
            _train_actor = (critic_warmup_remaining <= 0)
            metrics = main_agent.update(
                states_m, actions_m, lp_m,rets_arr, advs_arr,
                old_values=vals_m, batch_size=1024, epochs=4,
                train_actor=_train_actor
            )
            if critic_warmup_remaining > 0:
                critic_warmup_remaining -= 1
                if critic_warmup_remaining == 0:
                    print(f"🔥→✅ SICAK-BAŞLATMA bitti (ep {episode}): actor artık eğitiliyor, "
                          f"critic kalibre. critic_loss={metrics.get('critic_loss', 0):.3f}")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                main_agent.scheduler.step()
        # FIX 7: TensorBoard — genişletilmiş metrikler
        total_ep  = (episode - start_episode + 1) * NUM_PARALLEL_ENVS
        decided   = max(1, main_wins + opp_wins)
        frozen_total = max(1, main_wins + opp_wins + draws)
        # Kayan pencere win-rate (sadece frozen maçlar; min 20 sonuç)
        recent_wr = float(np.mean(recent_results)) if len(recent_results) >= 20 else 0.0

        writer.add_scalar('WinRate/Main_recent',   recent_wr * 100,             total_ep)
        writer.add_scalar('WinRate/Main_cum',      main_wins / decided * 100,   total_ep)
        writer.add_scalar('DrawRate',              draws / frozen_total * 100,  total_ep)
        writer.add_scalar('Curriculum/Difficulty', difficulty,                  total_ep)
        writer.add_scalar('Train/EntropyCoef',     metrics.get('ent_coef', 0.0), total_ep)
        writer.add_scalar('Train/AvgEpisodeLen',   float(np.mean(step_counts)), total_ep)
        writer.add_scalar('Train/LR',              main_agent.scheduler.get_last_lr()[0], total_ep)

        # Episode sonundaki HP ortalaması (her env'in son info'sundan hesaplanamıyor
        # çünkü reset edildi; step_counts üzerinden approximation)
        if rewards_m:
            writer.add_scalar('Rewards/Mean',    float(np.mean(rewards_m)),  total_ep)
            writer.add_scalar('Rewards/Std',     float(np.std(rewards_m)),   total_ep)

        if metrics:
            writer.add_scalar('Loss/Actor',      metrics.get('actor_loss',  0), total_ep)
            writer.add_scalar('Loss/Critic',     metrics.get('critic_loss', 0), total_ep)
            writer.add_scalar('Train/Entropy',   metrics.get('entropy',     0), total_ep)
            writer.add_scalar('Train/GradNorm',  metrics.get('grad_norm',   0), total_ep)
            writer.add_scalar('Train/KL',        metrics.get('kl',          0), total_ep)

        # ✅ EARLY STOPPING — kayan pencere win-rate ile
        # LOCOMOTION_PHASE'de KAPALI: P2 sabit/sıfır-aksiyon olduğu için
        # "kazanma oranı" anlamsız (hep %0) — bu metrikle asla "gelişme"
        # sayılmadığından patience her zaman dolup zararsız ama gereksiz bir
        # erken durdurmaya yol açıyordu (ep2790'da tam bu oldu, entropi
        # mükemmel yakınsamışken — eğitim durdu, çökme değildi).
        if not LOCOMOTION_PHASE and not GETUP_PHASE and not COMBINE_PHASE:   # frozen rakip → win-rate anlamsız
            current_win_rate = recent_wr
            if (episode - start_episode + 1) >= min_episodes_before_stop:
                if current_win_rate > best_win_rate:
                    best_win_rate = current_win_rate
                    no_improvement_count = 0
                    main_agent.save(best_ckpt_name)
                    print(f"💾 New best! Win rate: {best_win_rate*100:.1f}%")
                else:
                    no_improvement_count += 1

                if no_improvement_count >= patience:
                    print(f"\n🛑 Early stopping at episode {episode+1}")
                    print(f"   Best win rate: {best_win_rate*100:.1f}%")
                    break

        # ✅ Reporting
        if (episode + 1) % 10 == 0:
            draw_rate = draws / frozen_total * 100
            mode_str  = ("vs frozen" if use_frozen_this_ep
                         else "vs heuristic" if use_heuristic_this_ep
                         else "symmetric")
            live_ent_coef = float(main_agent.ent_coef_param.item())   # çift-yönlü (negatif olabilir)
            print(f"📊 Ep {episode+1}/{episodes} ({NUM_PARALLEL_ENVS}x) "
                  f"[Diff:{difficulty:.2f}] [{mode_str}] [ent_coef:{live_ent_coef:.4f}]")
            print(f"   🤖 Main : {main_wins} wins ({main_wins/decided*100:.1f}% küm. | "
                  f"{recent_wr*100:.1f}% son-{len(recent_results)})")
            print(f"   👻 Opp  : {opp_wins}  wins ({opp_wins/decided*100:.1f}%)")
            print(f"   ⚪ Draw : {draws} ({draw_rate:.1f}%)  [sadece frozen maçlar]")
            print(f"   ⚡ Steps: {np.mean(step_counts):.0f}  |  LR: {main_agent.scheduler.get_last_lr()[0]:.6f}")
            if rewards_m:
                print(f"   📈 Reward μ={np.mean(rewards_m):.3f} σ={np.std(rewards_m):.3f}")
            if metrics:
                sigma_now = float(torch.exp(main_agent.network.log_std_min +
                    0.5 * (main_agent.network.log_std_max - main_agent.network.log_std_min) *
                    (torch.tanh(main_agent.network.actor_log_std.detach()) + 1.0)).mean().item())
                print(f"   📉 actor={metrics.get('actor_loss',0):.4f}  "
                      f"critic={metrics.get('critic_loss',0):.4f}  "
                      f"ent={metrics.get('entropy',0):.4f}  "
                      f"ent_coef={metrics.get('ent_coef',0):.4f}  "
                      f"σ={sigma_now:.3f}  "
                      f"kl={metrics.get('kl',0):.5f}  "
                      f"‖g‖={metrics.get('grad_norm',0):.3f}")
            print()

        if (episode + 1) % save_freq == 0:
            main_agent.save(f"{ckpt_prefix}{episode+1}.pth")
            print(f"💾 Checkpoint: Episode {episode+1}\n")

    # ✅ Final save
    main_agent.save(final_ckpt_name)

    if parallel_envs:
        parallel_envs.close_all()
    writer.close()

    print("\n" + "="*70)
    print("✅ EĞİTİM TAMAMLANDI!")
    print("="*70)
    final_total  = (episode - start_episode + 1) * NUM_PARALLEL_ENVS
    decided_f    = max(1, main_wins + opp_wins)
    print(f"\n📊 FINAL İSTATİSTİKLER:")
    print(f"   Total episodes : {final_total}")
    print(f"   🤖 Main: {main_wins}/{decided_f} ({(main_wins/decided_f)*100:.1f}%)")
    print(f"   👻 Opp : {opp_wins}/{decided_f} ({(opp_wins/decided_f)*100:.1f}%)")
    print(f"   ⚪ Draw : {draws}/{final_total} ({(draws/final_total)*100:.1f}%)")
    print(f"   🏆 Best win rate: {best_win_rate*100:.1f}%")
    print("="*70 + "\n")
# TEST
# ============================================
def test_improved(xml_path="dual_humanoid_fight.xml", render_mode="human"):
    """Tek main_agent politikasını hem P1 hem P2 olarak oynatarak test eder."""
    print("\n🎬 SELF-PLAY TEST MATCH\n")

    import glob, re

    env = DualHumanoidFightEnv(render_mode=render_mode, xml_path=xml_path)
    main_agent = CUDAPPOAgent(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.p1_actuator_count,
        name="Main_Fighter"
    )

    def _ep_num(f):
        m = re.search(r'_ep(\d+)', f); return int(m.group(1)) if m else 0

    model_loaded = False
    for pattern, label in [
        ("main_agent_v2c_ep*.pth", "checkpoint"),
        (["best_main_agent_v2c.pth"], "best"),
        (["main_agent_v2c_final.pth"], "final"),
        ("main_agent_v2b_ep*.pth", "checkpoint(v2b)"),
        (["best_main_agent_v2b.pth"], "best(v2b)"),
        (["main_agent_v2b_final.pth"], "final(v2b)"),
    ]:
        if model_loaded:
            break
        files = sorted(glob.glob(pattern) if isinstance(pattern, str) else pattern,
                       key=_ep_num)
        if files and os.path.exists(files[-1]):
            try:
                print(f"📥 Yükleniyor: {files[-1]}")
                main_agent.load(files[-1])
                print(f"✓ {label} modeli yüklendi (ep {_ep_num(files[-1])})\n")
                model_loaded = True
            except Exception as e:
                print(f"⚠️  Yükleme hatası ({label}): {e}")

    if not model_loaded:
        print("⚠️  Hiçbir model bulunamadı — rastgele politikayla oynanıyor\n")

    env.reset()
    obs_p1, obs_p2 = env.get_obs_both()
    done = False
    step_count = 0

    print("🥊 MAÇ BAŞLIYOR...")
    print("="*60)

    while not done:
        # Aynı politika: P1 için own perspektifi, P2 için own perspektifi
        action1, _ = main_agent.get_action(obs_p1, training=False)
        action2, _ = main_agent.get_action(obs_p2, training=False)

        combined_action = np.concatenate([action1, action2])
        obs_pair, reward, terminated, truncated, info = env.step(combined_action)
        r1, r2 = reward['p1'], reward['p2']
        obs_p1, obs_p2 = env.get_obs_both()
        done = terminated or truncated

        env.render()
        time.sleep(0.016)
        step_count += 1

        if step_count % 100 == 0:
            combo1 = env.reward_system.combo_counter['p1']
            combo2 = env.reward_system.combo_counter['p2']
            print(f"⏱️  {step_count}: 🔴 {info['p1_hp']:.0f} HP (x{combo1}) | "
                  f"🔵 {info['p2_hp']:.0f} HP (x{combo2})")

    env.close()
    print("\n" + "="*60)
    print(f"🏆 {info['winner']}")
    print(f"⏱️  {step_count} steps")
    print(f"📊 Final: 🔴 {info['p1_hp']:.0f} HP | 🔵 {info['p2_hp']:.0f} HP")
    print(f"\n📈 STATS:")
    print(f"   P1: {info['stats']['p1_hits']} hits, "
          f"{info['stats']['p1_criticals']} crits, "
          f"max combo {info['stats']['p1_max_combo']}")
    print(f"   P2: {info['stats']['p2_hits']} hits, "
          f"{info['stats']['p2_criticals']} crits, "
          f"max combo {info['stats']['p2_max_combo']}")
    print("="*60 + "\n")

# ============================================
# AUTO-RESUME HELPER
# ============================================
def find_latest_checkpoint():
    """main_agent için en son YAZILAN checkpoint'in episode numarasını bul.

    Sadece v2 checkpoint'lere bakar (10 aktüatörlü kanıtlanmış mimari,
    dual_humanoid_fight.xml). v3 (14 aktüatörlü, dual_humanoid_fight_v2.xml)
    1200+ episode sonra hâlâ sistematik isabet üretemedi — proje v2'ye
    döndürüldü, v3 checkpoint'leri artık otomatik aranmıyor.

    ÖNEMLİ: dosya adındaki EN YÜKSEK episode numarası yerine dosyanın
    GERÇEK değişiklik zamanı (mtime) esas alınır. Bir regresyon sonrası
    daha eski bir episode'dan (örn ep950) resume edilirse, dosya sisteminde
    hâlâ duran ama artık geçersiz sayılan yüksek-numaralı bir checkpoint
    (örn ep1150, entropi-sürüklenmesiyle bozulmuş) YANLIŞLIKLA "en son"
    sanılıp tekrar yüklenmesin diye.
    """
    import glob, re
    # v2c: SADECE v2c checkpoint'lerine bakar. v2b (yürüyemeyen, "atıl ve düş"
    # çekim noktasına saplanmış) politikaları BİLİNÇLİ olarak aranmıyor —
    # yürüme-odaklı yeni ödül yapısıyla sıfırdan başlanıyor. Hiç v2c
    # checkpoint'i yoksa None döner → train_improved sıfırdan başlar.
    files = glob.glob("main_agent_v2c_ep*.pth")
    if not files:
        return None
    newest_file = max(files, key=os.path.getmtime)
    def _ep(f):
        m = re.search(r'_ep(\d+)', f); return int(m.group(1)) if m else 0
    latest_ep = _ep(newest_file)
    return latest_ep if os.path.exists(f"main_agent_v2c_ep{latest_ep}.pth") else None

if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    xml_path = "dual_humanoid_fight.xml"

    if mode == "train":
        # ✅ OTOMATIK RESUME: En son checkpoint'i bul
        latest_ep = find_latest_checkpoint()

        if latest_ep is not None:
            print(f"\n📥 En son checkpoint bulundu: Episode {latest_ep}")
            print(f"✅ Buradan devam ediliyor...\n")
            train_improved(episodes=5000, save_freq=50, xml_path=xml_path, resume_from=latest_ep)  # main_agent_ep{N}.pth
        else:
            print(f"\n🆕 Checkpoint bulunamadı, sıfırdan başlıyor...\n")
            train_improved(episodes=5000, save_freq=50, xml_path=xml_path)

        print("\n🎯 Test maçı başlatılıyor...\n")
        test_improved(xml_path=xml_path)

    elif mode == "test":
        test_improved(xml_path=xml_path)

    elif mode == "resume":
        # ✅ Manual resume mode
        resume_ep = int(sys.argv[2]) if len(sys.argv) > 2 else 150
        print(f"📥 Episode {resume_ep}'den devam ediliyor...")
        train_improved(episodes=5000, save_freq=50, xml_path=xml_path, resume_from=resume_ep)
        test_improved(xml_path=xml_path)

    elif mode == "quick":
        print("⚡ Hızlı test (50 episode)\n")
        train_improved(episodes=5000, save_freq=25, xml_path=xml_path)
        test_improved(xml_path=xml_path)

    print("✅ Tamamlandı!")