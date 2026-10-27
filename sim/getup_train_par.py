# -*- coding: utf-8 -*-
"""GETUP EGITIMI — PARALEL (multiprocessing) SURUM.

Neden gerekli: eski rollout ortamlari SIRAYLA kosuyordu -> 12 cekirdekli
i5-12500H'nin ~1 cekirdegi kullaniliyordu. Getup ~10M+ adim istiyor, biz 1.9M'deydik.

Kullanicinin onceki paralellestirme denemesi neden BASARISIZ oldu (olculdu):
  1) THREAD kullanilmisti -> GIL, CPU-bound MuJoCo'yu paralel kosturmaz.
     (selfplay_training.py sat.20'de bu not zaten var.)
  2) ADIM BASINA IPC tuzagi: bu makinede pipe gidis-donus = 83 us, fizik adimi
     = 46 us. Yani klasik SubprocVecEnv (her adimda worker'a gonder-al) ek yuku
     isin 1.8 KATI -> hizlanma degil YAVASLAMA.
  3) Windows 'spawn' tuzaklari: __main__ guard yoksa sonsuz surec uretimi;
     MuJoCo MjModel/MjData PICKLE EDILEMEZ (env worker'a gonderilemez);
     CUDA context surecler arasi tasinmaz.

COZUM (bu dosya):
  - Her worker TAM BIR EPISODE kosar (800 adim), yorungeyi TEK SEFERDE gonderir
    -> IPC episode basina 1 kez: ek yuk ~%0.2 (adim basina %180 yerine).
  - Worker env'i KENDI ICINDE kurar (pickle sorunu yok).
  - Worker'da politika CPU'da (st.DEVICE='cpu' set edilir) -> CUDA sorunu yok.
  - Kalici worker'lar (spawn maliyeti bir kez odenir).
  - Degerler (V) parent'ta TEK BATCH halinde GPU'da hesaplanir (adim basina degil).

Kullanim:
  python getup_train_par.py                # en guncel checkpoint'ten devam
  python getup_train_par.py --workers 6    # worker sayisi (varsayilan 6)
  python getup_train_par.py --sifirdan     # tabansiz
  python getup_train_par.py --ep-per-worker 4 --lr 2e-4   # update basina episode / lr
"""
import os, sys, glob, time, math
import multiprocessing as mp

import numpy as np

# ── varsayilanlar ────────────────────────────────────────────────────────
NUM_WORKERS_DEFAULT = 6      # kullanici tercihi: termal olarak guvenli bolge
CKPT_PREFIX  = "getup_artik_ep"  # 2026-10-17: PD + ARTIK eylem (taban = referans hareket), SIFIRDAN
# BECERI=yurume (ortam degiskeni, 2026-10-17): ayni paralel PPO ile YURUME (yurume_env.YurumeEnv). Ayarlanmazsa
# kalkis egitimi birebir eskisi gibi. Yurume: dusunce bolum biter (GAE son adimda bootstrap yok).
BECERI = os.environ.get("BECERI", "kalkis")
if BECERI == "dovus":   # 2026-10-19: dövüş hareketi uzmanı (dovus_env.DovusEnv; DOVUS_TIP=yumruk|tekme)
    CKPT_PREFIX = "dovus_" + os.environ.get("DOVUS_TIP", "yumruk") + "_ep"
if BECERI == "tek":     # 2026-10-18 plan B: TEK AĞ ince ayarı (tek_env.TekEnv; mutlak eylem, komut, itme, BC kaybı)
    CKPT_PREFIX = "tek_ep"
# 2026-10-21: ALT AĞIN DÖVÜŞE UYUMU (alt_dovus_env.AltDovusEnv; TEK_MODELI_v2'den, 109 gözlem). tek ile aynı ince ayar
# düzeni (lr, critic ısınması, donuk obs_rms, BC); BC verisi damitma2c_veri (109, zaten kilitli), simetri kapalı.
if BECERI == "alt_dovus":
    CKPT_PREFIX = "alt_dovus2_ep"     # v2 (2026-10-21): aktif rakip havuzu, nişan/yaklaşma ödülü, kısmi isabet (v1: alt_dovus_ep)
    if os.environ.get("ROBOT"):       # govde2 vb. (PLAN_EKLEMLER adım 7): ayrı kayıt adı, eski robotun kayıtlarına karışmasın
        CKPT_PREFIX = f"alt_dovus2_{os.environ['ROBOT']}_ep"
    if os.environ.get("KOSU"):        # 2026-10-22: aynı robotta ayrı koşu (ör. KOSU=belomuz) — önceki kayıtları ezmesin
        CKPT_PREFIX = CKPT_PREFIX[:-3] + "_" + os.environ["KOSU"] + "_ep"
TEK_BENZERI = BECERI in ("tek", "alt_dovus")
TEK_SYM_W = 0.25
TEK_LR = 1e-4            # 2e-4 ile ilk aktör güncellemeleri KL sınırını aşıyordu (büyük ağ, damıtılmış başlangıç)
BC_AZALMA_EP = 600       # BC ağırlığı 1.0 → BC_TABAN (ep 0..600, ~3M adım)
BC_TABAN = 0.5           # 2026-10-18: 0.1'e inince/kalkış az örneklenince yerden kalkış unutuldu (12'de 8 → 5 → 1)
BC_KALKIS_PAY = 0.5      # BC mini grubunun yarısı kalkış anlarından (gözlem[0] = baş z/2; ayakta eşiğinin altı)
BC_ORNEK = 1_000_000     # damitma_veri'den GPU'ya alınan öğretmen etiketi sayısı
if BECERI == "donme":   # 2026-10-18 B0: dönme yeteneği denemesi (yurume_env.DonmeEnv)
    CKPT_PREFIX = "donme_ep"
    if os.environ.get("DONME_YERINDE") == "1":   # 2026-10-19 Adım 3b: yerinde dönme, robota göre ayrı kayıt
        CKPT_PREFIX = "donme_yerinde_" + (os.environ.get("ROBOT") or "eski") + "_ep"
if BECERI == "yurume":
    CKPT_PREFIX = "yurume4_ep"   # 2026-10-17: + kalkis_sonu baslangici (yurume3: duz ayak odulu W_DUZ, 6M ayak -3.7°)
SAVE_FREQ    = 100            # ~480k adim; poz checkpoint'leri ayrica (POZ_EVERY)
# DURDURMA KRITERI — ONCEDEN YAZILDI (2026-10-13, kullanici + yayp-bilgisayar onayi):
# onceki kosular 0.5-2M adimda kesildi, bu erkendi. Bu koşu (yardim kuvvetli):
#   - en az MIN_STEPS (10M) ek adim; butce MAX_STEPS_DEFAULT (30M) -> kendiliginden durur
#   - BASARI = yerde VEYA oturma baslangicindan KUVVETSIZ ayakta bitis (POZ olcumu kuvvetsiz)
#   - kuvvet 0'a indikten sonra +4M adimda yerden kalkma hala <%10 ise -> FRASA planina
#     gecis onerilir (sagital+simetrik aksiyon, CrossQ/SAC). Logda ASSIST0 satiri bunu isaretler.
#   - her POZ_EVERY (2M) adimda deterministik poz dagilimi (mod basina POZ_N deneme)
#     loglanir ("POZ" satiri) + o anin checkpoint'i ..._poz{M}M.pth olarak saklanir.
# 20M'den once SADECE teknik ariza (NaN, coken KL/critic, mb=0) icin durdurulur.
MIN_STEPS         = 4_000_000   # k150: +4M sonra degerlendirme (kullanici onayi)
MAX_STEPS_DEFAULT = 30_000_000
POZ_EVERY         = 2_000_000
POZ_N             = 12
# YARDIM KUVVETI MUFREDATI (HoST tarzi): ~0.6*m*g ile basla; son ASSIST_WIN update'te
# yerde+oturma baslangiclarinin > ASSIST_RATE'i ayakta bittiyse ASSIST_STEP dus.
# Her azaltmadan sonra yeni pencere dolana kadar bekle (kuvvete 'asilma' kontrolu).
ASSIST_INIT  = 175.0        # N  (~0.85 m*g; m=21 kg). Plan 125 dedi (tahmin) ama olculdu:
#   10M politikasi stokastik, 12 deneme: 125N oturmadan 0/12; 150N 1/12 (yerde 3/12);
#   175N 6/12 (yerde 3/12); 200N 8/12. Baslangicta basari yoksa azaltma sinyali de yok.
ASSIST_STEP  = 12.5         # N  (%10)
ASSIST_WIN   = 20           # update
ASSIST_RATE  = 0.3          # 2026-10-13: %50 cok yuksekti (137.5 N'de ~900 update takildi)
HAVUZ_EPS    = 60           # havuz yenilemede kosulan episode
WARMUP_UPD   = 15
# UPDATE BASINA VERI (2026-10): eskiden worker basina 1 episode -> 6x200=1200
# ornek/update; ilk gradyan adiminda KL 0.2-0.5'e firliyordu (target 0.03).
# 4 episode -> 6x4x200=4800 ornek (Humanoid PPO icin olagan 4-5k mertebesi).
EPS_PER_WORKER_DEFAULT = 4
LR_DEFAULT   = 2e-4         # 5e-4 yerine (yurume fazinda LOCO_LR=2e-4 kararli calisti)
# KESIF GURULTUSU (2026-10, olculdu): dengede duran sabit poza aksiyon gurultusu
# eklenince ayakta kalma: sigma 0 -> 6.0 s (8/8), 0.05 -> 3.7 s, 0.1 -> 2.8 s,
# 0.2 -> 2.2 s (0/8). Egitim sigma=0.2 ile yapildigi icin ayakta durmayi deneyen
# her episode ~2.2 s'de dusuyordu -> critic "ayakta = dusmek" ogrendi, politika
# AYAKTA baslayip bile kendi istegiyle COMELIYORDU (odul degisikligi bunu kiramadi).
# Getup icin hedef sigma dusuk; ana kodun TARGET_SIGMA=0.20'si degismez.
GETUP_SIGMA  = 0.08
# KL BUTCESI (2026-10, yayp-bilgisayar olcumu): KL ~ (dmu/sigma)^2 -> sigma 0.2->0.08
# ayni adimda ~6x KL; sabit target_kl=0.03 ile tamamlanan mini-batch 11.7/20 ->
# 4.8/20 dustu (actor ~2.4x az ogreniyordu). sigma ile olcekle, tavan 0.1.
TARGET_KL    = min(0.1, 0.03 * (0.20 / GETUP_SIGMA) ** 2)


# ═════════════════════════════════════════════════════════════════════════
#  WORKER (ayri surec) — kendi env'i + CPU politikasi, tam episode kosar
# ═════════════════════════════════════════════════════════════════════════
def worker_loop(conn, worker_id, seed):
    """Kalici worker: agirlik al -> tam episode kos -> yorungeyi gonder."""
    import torch
    torch.set_num_threads(1)                 # asiri-abonelik olmasin
    np.random.seed(seed)
    torch.manual_seed(seed)

    import selfplay_training as st
    st.DEVICE = torch.device("cpu")          # KRITIK: worker'da CUDA YOK
    import getup_train as G                  # ayni env/odul tanimi

    if BECERI == "dovus":
        import dovus_env as DV
        env = DV.DovusEnv(env_id=100 + worker_id)
    elif BECERI == "tek":
        import tek_env as T
        env = T.TekEnv(env_id=100 + worker_id)
    elif BECERI == "alt_dovus":
        import alt_dovus_env as AD
        env = AD.AltDovusEnv(env_id=100 + worker_id)
    elif BECERI in ("yurume", "donme"):
        import yurume_env as Y
        env = (Y.DonmeEnv if BECERI == "donme" else Y.YurumeEnv)(env_id=100 + worker_id)
    else:
        env = G.GetupEnv(env_id=100 + worker_id)  # env WORKER ICINDE kurulur
    net = None

    while True:
        msg = conn.recv()
        if msg is None:
            break

        if net is None:
            net = st.CUDAPPONetwork(env.observation_space.shape[0],
                                    env.p1_actuator_count, gizli=msg.get("gizli"))
            net.eval()
        net.load_state_dict(msg["weights"])
        obs_mean, obs_var = msg["obs_mean"], msg["obs_var"]
        env.assist_force = float(msg.get("assist", 0.0))
        if "zorluk" in msg: env.zorluk = float(msg["zorluk"])
        conn.send([run_episode(env, net, obs_mean, obs_var, torch, G)
                   for _ in range(msg["n_eps"])])
    conn.close()


def run_episode(env, net, obs_mean, obs_var, torch, G):
    """Tek episode kos -> yorunge sozlugu (worker icinde)."""
    S, A, LP, R = [], [], [], []
    obs, _ = env.reset()
    done = False
    zmin, zmax = 9.9, 0.0
    while not done:
        nobs_in = (obs - obs_mean) / (np.sqrt(obs_var) + 1e-8)
        with torch.no_grad():
            t = torch.FloatTensor(nobs_in).unsqueeze(0)
            a_mean, a_std = net.eylem(t)      # critic yok (hız; aynı eylem)
            dist = torch.distributions.Normal(a_mean, a_std)
            # once KIRP, sonra log_prob: update() kirpilmis aksiyonun
            # log_prob'unu hesapliyor. Ters sira KL'i ~1.4 yapip her
            # update'i ilk mini-batch'te durduruyordu (actor hic egitilmedi).
            a = torch.clamp(dist.sample(), -1, 1)
            lp = dist.log_prob(a).sum(dim=-1)
        act = a.numpy()[0]
        S.append(obs); A.append(act); LP.append(float(lp.item()))
        obs, rew, term, trunc, _ = env.step(act)
        R.append(rew)
        z = float(env._get_player_position('p1')[2])
        zmin = min(zmin, z); zmax = max(zmax, z)
        done = term or trunc
    son_term = bool(term)

    z = float(env._get_player_position('p1')[2])
    tu = float(env.data.xmat[env.p1_torso_id][8])
    ends_up = z > G.STAND_Z and tu > G.STAND_TORSO
    return {
        "states":  np.asarray(S, np.float32),
        "actions": np.asarray(A, np.float32),
        "logps":   np.asarray(LP, np.float32),
        "rewards": np.asarray(R, np.float32),
        "last_obs": obs.astype(np.float32),
        "ends_up": bool(ends_up),
        "fell":    bool(zmin < 0.40 + G.GOVDE_EK),
        "launch":  bool(zmax > 1.3),
        "ep_ret":  float(np.sum(R)),
        "mode":    env.start_mode,          # itme / yerde / rsi / oturma
        "assist_frac": env.assist_steps / max(len(R), 1),
        "mimic_on":  bool(env.mimic_on),
        "mimic_r":   float(env.mimic_sum) / max(len(R), 1),
        "mimic_rel": bool(env.mimic_released),
        "tuck_sum":  float(env.tuck_sum),
        "yvideo":    env.start_mode == "yerde" and getattr(env, "_yerde_video", None) is not None,
        "otur_src":  env.oturma_src if env.start_mode in ("oturma", "toplama") else None,
        "terminated": son_term,
        "mesafe":    float(env.mesafe()) if hasattr(env, "mesafe") else 0.0,
        "sure_s":    len(R) * G.PHYS_DT * G.FRAME_SKIP,
        "ayakta_oran": getattr(env, "ayakta_adim", 0) / max(len(R), 1),
        "yon_hata": (np.degrees(env.yon_hata_top / env.yon_hata_n) if getattr(env, "yon_hata_n", 0) else None),
    }


# ═════════════════════════════════════════════════════════════════════════
#  PARENT
# ═════════════════════════════════════════════════════════════════════════
def main():
    import torch
    import selfplay_training as st
    import getup_train as G

    def argval(name, default):
        if name in sys.argv:
            i = sys.argv.index(name)
            if i + 1 < len(sys.argv):
                return sys.argv[i + 1]
        return default

    n_workers = int(argval("--workers", NUM_WORKERS_DEFAULT))
    eps_per_worker = int(argval("--ep-per-worker", EPS_PER_WORKER_DEFAULT))
    lr = float(argval("--lr", TEK_LR if TEK_BENZERI else LR_DEFAULT))
    scratch = "--sifirdan" in sys.argv
    resume = argval("--resume", None)
    if resume is None and not scratch:
        # SADECE bu koşunun checkpoint'leri: eski getup_solo_*/getup_yer_*/getup_k120_*
        # ESKI FIZIK (bilek k=30) — sessizce onlara dusmek hataydi (yayp incelemesi).
        cks = sorted(glob.glob(CKPT_PREFIX + "*.pth"), key=os.path.getmtime)
        if not cks:
            sys.exit(f"! {CKPT_PREFIX}*.pth yok — --resume <ckpt> veya --sifirdan ver.")
        resume = cks[-1]

    # parent env: sadece boyutlari ogrenmek + degerlendirme icin
    if BECERI == "dovus":
        import dovus_env as DV, yurume_env as Y
        probe = DV.DovusEnv(env_id=0)
    elif BECERI == "tek":
        import tek_env as T, yurume_env as Y
        probe = T.TekEnv(env_id=0)
    elif BECERI == "alt_dovus":
        import tek_env as T, yurume_env as Y, alt_dovus_env as AD
        probe = AD.AltDovusEnv(env_id=0)
    elif BECERI in ("yurume", "donme"):
        import yurume_env as Y
        probe = (Y.DonmeEnv if BECERI == "donme" else Y.YurumeEnv)(env_id=0)
    else:
        probe = G.GetupEnv(env_id=0)
    obs_dim = probe.observation_space.shape[0]
    act_dim = probe.p1_actuator_count

    # adaptif entropi hedefi bu koşu icin GETUP_SIGMA (agent __init__ okur)
    st.TARGET_SIGMA = GETUP_SIGMA
    st.TARGET_ENTROPY = 0.5 * math.log(2 * math.pi * math.e) + math.log(GETUP_SIGMA)
    agent = st.CUDAPPOAgent(obs_dim, act_dim, name="GetupPar")
    # SOL/SAG SIMETRI KAYBI (2026-10-14): bkz. getup_train.ayna_haritalari / SYM_W
    agent.sym = (*(Y.yurume_ayna_haritalari() if BECERI == "yurume" else G.ayna_haritalari()), G.SYM_W)
    if BECERI in ("donme", "dovus", "alt_dovus"):
        agent.sym = None                    # tek yöne dönme / taraflı vuruş referansı ayna simetrisiyle çelişir
    if BECERI == "tek":
        # SYM_W 4 → 0.25 (2026-10-18, ölçüldü): öğrenci simetri kaybıyla damıtılmadığı için aktör eğitimi başlar başlamaz
        # sim kaybı 0.094 → ilk mini grupta KL 0.23 (hedef 0.1) ve ayakta %80 → %44. Plan gereği açık, ama küçük.
        agent.sym = (*T.tek_ayna_haritalari(), TEK_SYM_W)
    print(f"  simetri kaybi: agirlik {G.SYM_W}", flush=True)
    warmup = WARMUP_UPD
    zorluk, win_ayakta, win_yon = 0.0, [], []   # BECERI=tek müfredatı: ayakta > 0.75 VE yön hatası < eşik → zorluk +0.1
    if scratch:
        print("SIFIRDAN egitim (taban yok)", flush=True)
        warmup = 0
        # sigma'yi TARGET_SIGMA'dan baslat (ag varsayilani ~0.6 = asiri gurultu);
        # ana egitimdeki (train_improved) ayni formul, sonrasi adaptif entropi.
        net = agent.network
        with torch.no_grad():
            raw = math.atanh(2 * (math.log(st.TARGET_SIGMA) - net.log_std_min) /
                             (net.log_std_max - net.log_std_min) - 1)
            net.actor_log_std.fill_(raw)
    elif os.path.exists(resume):
        agent.load(resume)
        print(f"Taban yuklendi: {resume}", flush=True)
    else:
        print(f"! Taban yok ({resume}) — sifirdan", flush=True)
    # sigma'yi hedefe cek (yuklenen checkpoint 0.20'de olabilir; adaptif entropi
    # tek basina cok yavas iner). Formul ana egitimdekiyle ayni.
    net = agent.network
    with torch.no_grad():
        raw = math.atanh(2 * (math.log(GETUP_SIGMA) - net.log_std_min) /
                         (net.log_std_max - net.log_std_min) - 1)
        net.actor_log_std.fill_(raw)
    print(f"  sigma hedefi={GETUP_SIGMA} (adaptif entropi hedefi ayni), target_kl={TARGET_KL:.3f}", flush=True)
    YENI_SIGMA = float(os.environ.get("YENI_SIGMA", "0"))   # 2026-10-22: yeni eklemlerde (12+) daha geniş keşif
    if YENI_SIGMA > 0 and act_dim > 12:
        with torch.no_grad():
            raw = math.atanh(2 * (math.log(YENI_SIGMA) - net.log_std_min) / (net.log_std_max - net.log_std_min) - 1)
            net.actor_log_std[12:] = raw
        print(f"  yeni eklem sigmasi={YENI_SIGMA} (eylem 12..{act_dim - 1})", flush=True)
    if TEK_BENZERI:
        # critic damıtmada hiç eğitilmedi (rastgele) → uzun sıcak başlatma (aktör donuk)
        warmup = max(warmup, 40)
        # BC ek kaybı: damitma_veri'den öğretmen etiketleri (gözlem 103 → 105: yön hatası (0, 1) = 'hedef tam önde')
        # alt_dovus: damitma2c_veri (109 gözlem; kilitli + komut/vuruş boyutları hazır)
        import glob as _g
        # BELLEK: tüm veriyi (~3 GB) yükleyip seçmek yerine her dosyadan orantılı alt örnek (2026-10-18)
        bc_dizin = "damitma2c_veri" if BECERI == "alt_dovus" else "damitma_veri"
        dosyalar = sorted(_g.glob(f"{bc_dizin}/tur*.npz")); boy = [len(np.load(f)["L"]) for f in dosyalar]
        rng = np.random.default_rng(0); Os, Ls = [], []
        for f, nb in zip(dosyalar, boy):
            v = np.load(f); k = int(round(BC_ORNEK * nb / sum(boy))); i = rng.choice(nb, min(k, nb), replace=False)
            Os.append(v["O"][i].astype(np.float32)); Ls.append(v["L"][i]); del v
        O = np.concatenate(Os); L = np.concatenate(Ls); del Os, Ls; sec = np.arange(len(O))
        if BECERI == "tek":
            O = np.concatenate([T.gozlem_kilitle(O), np.tile([[0.0, 1.0]], (len(O), 1))], 1).astype(np.float32)
        bc_eylem = None
        if BECERI == "alt_dovus" and obs_dim != O.shape[1]:
            O, L = T.bc_robota_uyarla(O, L)   # govde2 (PLAN_EKLEMLER adım 5): yeni eklemler nötr gözlem + nötr etiket
            # 2026-10-22 (ölçüldü: yeni eklemler ort 1–2°): nötr etiket BC kaybıyla yeni eklemleri NÖTRDE TUTUYORDU →
            # BC yalnız eski 12 eylemde; yeniler taklit (bel/omuz referansı) + dövüş ödülüyle öğrenilir
            bc_eylem = 12
            print(f"  BC verisi bu robota uyarlandi: gozlem {O.shape[1]}, eylem {L.shape[1]} (BC yalniz ilk {bc_eylem} eylem)", flush=True)
        if BECERI == "alt_dovus" and obs_dim > O.shape[1]:
            # 2026-10-25 RAKIP_GOZ: damıtma verisinde rakip bilgisi yok → yeni (sondaki) boyutlar 0
            O = np.concatenate([O, np.zeros((len(O), obs_dim - O.shape[1]), O.dtype)], 1)
            print(f"  BC verisi rakip-gozlemi icin {obs_dim} boyuta 0 ile dolduruldu", flush=True)
        agent.obs_rms_sabit = True        # damıtılmış ağın normalizasyonu donuk (ilk koşu: güncellenen rms → KL 8.2)
        kalkis = np.where(O[:, 0] * 2.0 < G.STAND_Z)[0]          # baş yüksekliği ayakta eşiğinin altında
        agent.bc = dict(O=torch.as_tensor(O, device=st.DEVICE), L=torch.as_tensor(L, device=st.DEVICE), w=1.0, n=512,
                        kalkis_idx=torch.as_tensor(kalkis, device=st.DEVICE), kalkis_pay=BC_KALKIS_PAY,
                        eylem_sayisi=bc_eylem if TEK_BENZERI else None)
        print(f"  BC kalkis anlari: {len(kalkis):,} / {len(O):,} (%{100 * len(kalkis) / len(O):.0f}) -> mini grubun %{100 * BC_KALKIS_PAY:.0f}'i", flush=True)
        del O, L
        print(f"  BC ek kaybi: {len(sec):,} ogretmen etiketi, agirlik 1.0 -> {BC_TABAN} (ep {BC_AZALMA_EP})", flush=True)
    # load() optimizer state'iyle eski lr'yi (5e-4) geri getirir -> sonra ez
    for pg in agent.optimizer.param_groups:
        pg["lr"] = lr

    # ── worker'lari kur (spawn maliyeti BIR KEZ) ──
    ctx = mp.get_context("spawn")
    conns, procs = [], []
    for w in range(n_workers):
        pc, cc = ctx.Pipe()
        p = ctx.Process(target=worker_loop, args=(cc, w, 1234 + w * 77), daemon=True)
        p.start()
        conns.append(pc); procs.append(p)
    print(f"PARALEL GETUP: {n_workers} worker x {eps_per_worker} episode/update "
          f"(~{n_workers * eps_per_worker * G.MAX_STEPS} ornek/update), lr={lr:g}", flush=True)
    print(f"  steps={G.MAX_STEPS} itme=uniform({G.PUSH_MIN},{G.PUSH_MAX})", flush=True)

    ep = 0
    total_steps, next_poz = 0, POZ_EVERY
    max_steps = int(float(argval("--max-adim", MAX_STEPS_DEFAULT)))
    print(f"  adim butcesi={max_steps:,} (min {MIN_STEPS:,}), poz olcumu her {POZ_EVERY:,}", flush=True)
    win_yer = []                       # son 20 update: (yerden kalkan, yerde baslayan)
    assist = float(argval("--kuvvet", ASSIST_INIT))
    win_assist, last_assist_change, assist_zero_step = [], 0, None
    print(f"  yardim kuvveti={assist:.1f} N (adim {ASSIST_STEP}, pencere {ASSIST_WIN}, esik {ASSIST_RATE})", flush=True)
    if BECERI in ("yurume", "donme"):
        print(f"  BECERI={BECERI} baslangic={Y.YURUME_START} bolum {Y.YURUME_SURE_S} s, gozlem {obs_dim}", flush=True)
    elif BECERI == "dovus":
        print(f"  BECERI=dovus tip={DV.DOVUS_TIP} referans {len(probe._R)} baslangic={DV.DOVUS_START} gozlem {obs_dim}", flush=True)
    elif BECERI == "alt_dovus":
        print(f"  BECERI=alt_dovus bolum {AD.SURE_S} s, rakip donuk TEK_MODELI_v2, gozlem {obs_dim}", flush=True)
    elif BECERI == "tek":
        # DIKKAT (2026-10-18 hatası, Ders 17): havuz_yenile YALNIZ kalkış eğitimi içindir — tek koşusunda çalışınca
        # oturma_havuzu.npz'yi (kalkış testleri/eğitimleri kullanır) başka bir ajanın durumlarıyla EZDİ.
        print(f"  BECERI=tek baslangic={T.TEK_START} bolum {T.TEK_SURE_S} s, gozlem {obs_dim}", flush=True)
    else:
        print(f"  oturma havuzu: {havuz_yenile(agent, probe, torch, G)}", flush=True)
        print(f"  baslangic={G.START_PROBS}", flush=True)
    try:
        while True:
            ep += 1
            t0 = time.time()

            # 1) agirliklari CPU'ya kopyalayip tum worker'lara gonder
            w_cpu = {k: v.detach().cpu() for k, v in agent.network.state_dict().items()}
            payload = {"weights": w_cpu, "gizli": agent.network.gizli, "zorluk": zorluk,
                       "obs_mean": agent.obs_rms.mean.copy(),
                       "obs_var":  agent.obs_rms.var.copy(),
                       "n_eps":    eps_per_worker,
                       "assist":   assist}
            for c in conns:
                c.send(payload)

            # 2) yorungeleri topla (worker'lar PARALEL kosuyor)
            traj = [t for c in conns for t in c.recv()]   # worker -> episode listesi
            t_roll = time.time() - t0

            # 3) DEGERLERI TEK BATCH'TE hesapla (adim basina degil!)
            all_states = np.concatenate([t["states"] for t in traj], axis=0)
            last_obs = np.stack([t["last_obs"] for t in traj], axis=0)
            values_all = batched_values(agent, all_states, torch)
            last_vals = batched_values(agent, last_obs, torch)

            # 4) GAE (her yorunge ayri; hepsi truncated -> critic bootstrap)
            states, actions, logps, rets, advs, vals = [], [], [], [], [], []
            off = 0
            for i, t in enumerate(traj):
                n = len(t["states"])
                V = values_all[off:off + n].tolist(); off += n
                dones = [0] * n
                if t.get("terminated"):        # yurume: dustu -> son adimda bootstrap yok
                    dones[-1] = 1
                adv = agent.compute_gae(t["rewards"].tolist(), V,
                                        float(last_vals[i]), dones)
                ret = [a + v for a, v in zip(adv, V)]
                states.extend(t["states"]); actions.extend(t["actions"])
                logps.extend(t["logps"].tolist())
                rets.extend(ret); advs.extend(adv); vals.extend(V)

            advs_arr = np.array(advs, dtype=np.float32)
            if len(advs_arr) > 1:
                advs_arr = (advs_arr - advs_arr.mean()) / (advs_arr.std() + 1e-8)
            if getattr(agent, "bc", None) is not None:
                agent.bc["w"] = max(BC_TABAN, 1.0 - (1.0 - BC_TABAN) * ep / BC_AZALMA_EP)
            info = agent.update(states, actions, logps, np.array(rets, np.float32),
                         advs_arr, old_values=vals, batch_size=1024, epochs=4,
                         train_actor=(warmup <= 0), target_kl=TARGET_KL)
            if warmup > 0:
                warmup -= 1
                if warmup == 0:
                    print(f"  [ep{ep}] critic kalibre — actor egitiliyor", flush=True)

            if BECERI == "tek":
                win_ayakta.append(np.mean([t["ayakta_oran"] for t in traj])); win_ayakta = win_ayakta[-10:]
                yh = [t["yon_hata"] for t in traj if t.get("yon_hata") is not None]
                win_yon.append(np.mean(yh) if yh else 0.0); win_yon = win_yon[-10:]
                # 2026-10-19: zorluk yalnız ayakta kalmaya değil KOMUT TAKİBİNE de bağlı (yürürken ort. |Δψ| < YON_ESIK_DER)
                if (warmup <= 0 and len(win_ayakta) == 10 and np.mean(win_ayakta) > 0.75 and zorluk < 1.0
                        and np.mean(win_yon) < T.YON_ESIK_DER):
                    zorluk = round(min(1.0, zorluk + 0.1), 2); win_ayakta = []; win_yon = []
                    print(f"  [ep{ep}] MUFREDAT: zorluk -> {zorluk}", flush=True)
            # 5) istatistik
            stand = sum(t["ends_up"] for t in traj)
            fell = sum(t["fell"] for t in traj)
            rec = sum(t["ends_up"] and t["fell"] for t in traj)
            launch = sum(t["launch"] for t in traj)
            ret_mean = float(np.mean([t["ep_ret"] for t in traj]))
            steps = len(all_states)

            if ep % 5 == 0 or ep == 1:
                n_eps = len(traj)
                with torch.no_grad():
                    sigma = float(agent.network(torch.zeros(1, obs_dim, device=st.DEVICE))[1].mean())
                print(f"ep{ep:5d} ayakta={stand/n_eps*100:.0f}% "
                      f"dusen={fell}/{n_eps} kalkan={rec}/{max(fell,1)} "
                      f"firlama={launch} odul={ret_mean:.0f} "
                      f"| sigma={sigma:.3f} ent={info['ent_coef']:.4f} "
                      f"kl={info['kl']:.3f} mb={info['n_steps']}/{4 * -(-steps // 1024)} critic={info['critic_loss']:.3f} sim={info.get('sym_loss', 0):.4f} bc={info.get('bc_loss', 0):.4f} zor={zorluk:.1f} ay%={100 * np.mean([t.get('ayakta_oran', 0) for t in traj]):.0f} yonH={np.mean([t['yon_hata'] for t in traj if t.get('yon_hata') is not None] or [0]):.0f} "
                      f"| {steps/ (time.time()-t0):.0f} adim/sn "
                      f"(rollout {t_roll:.1f}s)", flush=True)
            # moda gore: yerde baslayip episode sonunda ayakta = GERCEK kalkma
            by = {m: [t["ends_up"] for t in traj if t["mode"] == m]
                  for m in ("itme", "yerde", "rsi", "oturma", "toplama", "zincir")}
            yer_up, yer_n = sum(by["yerde"]), len(by["yerde"])
            win_yer.append((yer_up, yer_n)); win_yer[:] = win_yer[-20:]
            if ep % 5 == 0 or ep == 1:
                wu, wn = sum(a for a, _ in win_yer), sum(b for _, b in win_yer)
                af = float(np.mean([t['assist_frac'] for t in traj]))
                print(f"        mod: itme_ayakta={sum(by['itme'])}/{len(by['itme'])} "
                      f"YERDEN_KALKAN={yer_up}/{yer_n} OTURMADAN={sum(by['oturma'])}/{len(by['oturma'])} "
                      f"rsi_ayakta={sum(by['rsi'])}/{len(by['rsi'])} "
                      f"| son20 yerden-kalkma={100*wu/max(wn,1):.1f}% | kuvvet={assist:.1f}N aktif%={100*af:.0f}", flush=True)
                mt = [t for t in traj if t['mimic_on']]
                ot = [t for t in traj if t['mode'] == 'oturma']
                oh = [t for t in ot if t['otur_src'] == 'havuz']; ov = [t for t in ot if t['otur_src'] == 'video']
                print(f"        taklit: aktif bolum={len(mt)}/{len(traj)} r_im/adim ort="
                      f"{np.mean([t['mimic_r'] for t in mt]) if mt else 0:.2f} (maks 2.0) | "
                      f"oturma-havuz: TOPLADI={sum(t['mimic_rel'] for t in oh)}/{len(oh)} "
                      f"(toplama F ort={np.mean([t['tuck_sum'] for t in oh]) if oh else 0:+.2f}) "
                      f"ayakta={sum(t['ends_up'] for t in oh)} | oturma-video(diz cekili): "
                      f"ayakta={sum(t['ends_up'] for t in ov)}/{len(ov)}", flush=True)
                yv = [t for t in traj if t['yvideo']]; yo = [t for t in traj if t['mode'] == 'yerde' and not t['yvideo']]
                print(f"        yerde: video-basl. ayakta={sum(t['ends_up'] for t in yv)}/{len(yv)} "
                      f"(taklit r/adim {np.mean([t['mimic_r'] for t in yv]) if yv else 0:.2f}/5.0) | "
                      f"eski-basl. ayakta={sum(t['ends_up'] for t in yo)}/{len(yo)}", flush=True)
                zc = [t for t in traj if t['mode'] == 'zincir']
                print(f"        zincir: ayakta={sum(t['ends_up'] for t in zc)}/{len(zc)} "
                      f"(taklit r/adim {np.mean([t['mimic_r'] for t in zc]) if zc else 0:.2f}/5.0)", flush=True)
                tp = [t for t in traj if t['mode'] == 'toplama']
                print(f"        toplama-modu: TOPLADI={sum(t['mimic_rel'] for t in tp)}/{len(tp)} "
                      f"ayakta={sum(t['ends_up'] for t in tp)}/{len(tp)}", flush=True)
            if yer_up > 0:
                print(f"KALKMA ep{ep}: yerden {yer_up}/{yer_n}", flush=True)
            # yardim kuvveti mufredati: yerde+oturma baslangiclarindan (kuvvetli) ayakta bitis
            g_up = sum(by["yerde"]) + sum(by["oturma"]); g_n = len(by["yerde"]) + len(by["oturma"])
            win_assist.append((g_up, g_n)); win_assist[:] = win_assist[-ASSIST_WIN:]
            if assist > 0 and ep - last_assist_change >= ASSIST_WIN and len(win_assist) >= ASSIST_WIN:
                ru = sum(a for a, _ in win_assist) / max(sum(b for _, b in win_assist), 1)
                if ru > ASSIST_RATE:
                    assist = max(0.0, assist - ASSIST_STEP)
                    last_assist_change = ep; win_assist.clear()
                    print(f"KUVVET ep{ep}: kalkma(kuvvetli) {100*ru:.0f}% > {100*ASSIST_RATE:.0f}% "
                          f"-> {assist:.1f} N", flush=True)
                    if assist == 0.0 and assist_zero_step is None:
                        assist_zero_step = total_steps
                        print(f"ASSIST0 kuvvet sifira indi @ {total_steps:,} adim — FRASA olcutu: "
                              f"+4M adimda kuvvetsiz yerden kalkma <%10 ise gecis onerilir", flush=True)
            if ep % SAVE_FREQ == 0:
                agent.save(f"{CKPT_PREFIX}par{ep}.pth")

            # adim butcesi + periyodik poz olcumu (bkz. DURDURMA KRITERI)
            total_steps += steps
            if total_steps >= next_poz:
                M = total_steps // 1_000_000
                agent.save(f"{CKPT_PREFIX}poz{M}M.pth")
                print(f"POZ {M}M adim (ep{ep}) [KUVVETSIZ; egitim kuvveti={assist:.1f}N]: "
                      f"{DV.dovus_olcum(agent, probe) if BECERI == 'dovus' else AD.alt_dovus_olcum(agent, probe) if BECERI == 'alt_dovus' else T.tek_olcum(agent, probe) if BECERI == 'tek' else yurume_olcum(agent, probe) if BECERI in ('yurume', 'donme') else poz_dagilimi(agent, probe, torch, G)}", flush=True)
                if BECERI not in ("yurume", "donme", "tek", "dovus", "alt_dovus"):
                    print(f"  oturma havuzu: {havuz_yenile(agent, probe, torch, G)}", flush=True)
                next_poz += POZ_EVERY
            if total_steps >= max_steps:
                agent.save(f"{CKPT_PREFIX}son{total_steps // 1_000_000}M.pth")
                print(f"BUTCE DOLDU: {total_steps:,} adim — egitim bitti.", flush=True)
                break
    finally:
        for c in conns:
            try: c.send(None)
            except Exception: pass
        for p in procs:
            p.join(timeout=3)
            if p.is_alive(): p.terminate()


def yurume_olcum(agent, env, N=12):
    """Yurume POZ'u: deterministik, kuvvetsiz; baslangic basina N deneme -> 10 s tamamlayan, ort. sure/mesafe/hiz/yan."""
    parca = []
    for mode in ("ayakta", "yurume_rsi", "kalkis_sonu"):
        sure, mes, yan, tam, don = [], [], [], 0, []
        for t in range(N):
            np.random.seed(20_000 + t); o, _ = env.reset(options={"start": mode}); done = False; wz = []
            while not done:
                a, _ = agent.get_action(o, training=False)
                o, _r, term, trunc, _ = env.step(a); done = term or trunc
                if hasattr(env, "wz_son"): wz.append(env.wz_son)
            if wz: don.append(np.degrees(np.mean(wz)))
            s_ = env.timestep * env.model.opt.timestep * 2; sure.append(s_); mes.append(env.mesafe()); yan.append(abs(env.yan_sapma()))
            tam += bool(trunc and not term)
        parca.append(f"{mode}[10s_tamam={tam}/{N} sure={np.mean(sure):.1f}s mesafe={np.mean(mes):.2f}m "
                     f"hiz={np.sum(mes)/max(np.sum(sure),1e-9):.2f}m/s yan={np.mean(yan):.2f}m"
                     + (f" donme={np.mean(don):+.1f}deg/s (min {np.min(don):+.1f})" if don else "") + "]")
    np.random.seed(None)
    return "  ".join(parca)


def poz_dagilimi(agent, env, torch, G):
    """Deterministik politika, her baslangic modunda POZ_N deneme -> son poz sinifi.
    yatik / oturma / comelme / ayakta (bas z ve govde dikligi; degerlendirme/ ile ayni esikler)."""
    import selfplay_training as st
    def sinif(z, tu):
        if z > G.STAND_Z and tu > G.STAND_TORSO: return "ayakta"
        if z >= 0.42 + G.GOVDE_EK and tu > 0.6: return "comelme"
        if tu > 0.6: return "oturma"
        return "yatik"
    parcalar = []
    env.assist_force = 0.0                   # olcum HER ZAMAN kuvvetsiz
    for mode in ("itme", "rsi", "yerde", "oturma", "toplama", "zincir"):
        c = {}
        for t in range(POZ_N):
            np.random.seed(10_000 + t)
            o, _ = env.reset(options={"start": mode}); done = False
            while not done:
                a, _ = agent.get_action(o, training=False)
                o, _r, term, trunc, _ = env.step(a); done = term or trunc
            k = sinif(float(env._get_player_position('p1')[2]),
                      float(env.data.xmat[env.p1_torso_id][8]))
            c[k] = c.get(k, 0) + 1
        parcalar.append(f"{mode}[" + " ".join(f"{k}={c.get(k, 0)}" for k in
                        ("ayakta", "comelme", "oturma", "yatik")) + "]")
    np.random.seed(None)
    return "  ".join(parcalar)


def havuz_yenile(agent, env, torch, G):
    """Guncel politikayla (stokastik, KUVVETSIZ) yerde/oturma baslangiclarindan
    HAVUZ_EPS episode kos; sonu dik-ama-alcak (oturma/comelme) olan durumlari
    G.HAVUZ_FILE dosyasina yaz (atomik). Ters mufredat: politika ilerledikce havuz yukselir."""
    env.assist_force = 0.0
    p1s, d = env.p1_qpos_start, env.p1_root_dof_adr
    Q, V = [], []
    for t in range(HAVUZ_EPS):
        mode = "yerde" if t % 2 == 0 else "oturma"
        o, _ = env.reset(options={"start": mode}); done = False
        while not done:
            a, _ = agent.get_action(o, training=True)
            o, _r, term, trunc, _ = env.step(a); done = term or trunc
        z = float(env._get_player_position('p1')[2]); tu = float(env.data.xmat[env.p1_torso_id][8])
        if tu > G.ASSIST_TORSO and z < G.ASSIST_HEAD_Z:
            Q.append(env.data.qpos[p1s:p1s + 19].copy()); V.append(env.data.qvel[d:d + 18].copy())
    if len(Q) < 10:
        return f"yenilenmedi (yalniz {len(Q)} oturma durumu; eski havuz korunuyor)"
    tmp = G.HAVUZ_FILE + ".tmp.npz"
    np.savez(tmp, qpos=np.array(Q), qvel=np.array(V))
    os.replace(tmp, G.HAVUZ_FILE)
    zs = [q[2] for q in Q]
    return f"{len(Q)}/{HAVUZ_EPS} durum, kok z ort={np.mean(zs):.2f} -> {G.HAVUZ_FILE}"


def batched_values(agent, obs_batch, torch):
    """Tum gozlemler icin V=min(V1,V2) TEK GPU cagrisinda (adim basina degil)."""
    import selfplay_training as st
    x = agent.obs_rms.normalize(np.asarray(obs_batch, np.float32))
    out = []
    with torch.no_grad():
        for i in range(0, len(x), 4096):
            t = torch.FloatTensor(x[i:i + 4096]).to(st.DEVICE)
            _m, _s, v1, v2 = agent.network(t)
            out.append(torch.min(v1, v2).cpu().numpy())
    v = np.concatenate(out) if out else np.zeros(0, np.float32)
    m, s = agent._ret_mean_std()
    return v * s + m                       # deger-norm geri al (raw olcek)


if __name__ == "__main__":           # KRITIK: Windows spawn guard
    mp.freeze_support()
    main()
