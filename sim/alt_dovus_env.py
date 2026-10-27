# -*- coding: utf-8 -*-
"""Alt ağın DÖVÜŞE UYUM ince ayarı ortamı (2026-10-21; plan F3 öne alındı — kullanıcı onayı 'devam et burdan').
Duman testleri (dovus_mac_env, donuk TEK_MODELI_v2): rakibe dönemiyor (yürürken sağa sapıp uzaklaşıyor), rakip gövdesine
yakın vuruşta isabet ~0 ve sık düşme → taktik ağdan ÖNCE alt ağ rakip karşısında eğitilmeli.

v1 sonucu (bulut 6M, alt_dovus_epson6M): rakibe açı 158°→32°, mesafe 5.85→0.88 m, ayakta %98, isabet 0/8. Teşhis: ~0.8 m'de
  duruyordu (vuruş eşiği 0.62 m, yumruk menzili ~0.45 m) ve vuruşlar rakibin önde duran KOLLARINA çarpıyordu → ödül hiç gelmedi.
v2 (2026-10-21, kullanıcı onayı):
  * vuruş eşiği YAKLAS_M 0.45 m · yaklaşırken mesafe kısaltma ödülü (W_YAKLAS)
  * kısmi isabet (dovus_mac_env: gard ×0.25, uyluk ×0.5; baş ×2, göğüs ×1), isabet ödülü = W_ISABET × hasar
  * NİŞAN: vuruş anı penceresinde (olay ±0.25 s) vuran uç ↔ rakip baş/göğüs mesafesi (yoğun ödül; isabetsiz de yön verir)
  * AKTİF RAKİP (kullanıcı isteği): mavi de aynı senaryoyla yaklaşıp vurur; ağı RAKIP_HAVUZ'dan (geçmiş en iyi kayıtlar)
    bölüm başına rastgele. PASIF_P oranında pasif kalır (kolay başlangıç). Alınan hasar cezası (W_ALINAN).
  * yumruk/tekme referansı DOVUS_HIZ ile hızlandırılabilir (1.5: TEK_MODELI_v2 yumrukta 0/6 → 3/6 isabet, düşme 4/6 → 1/6)
  Öğrenenin gözleminde rakip boyutları KİLİTLİ (alt ağ değişmesin): rakibin vuruşunu göremez, yalnız tepki/denge öğrenir.

P1 = ÖĞRENEN (kırmızı; 109 gözlem, mutlak eylem — TEK_MODELI_v2 ile aynı), P2 = rakip (mavi; donuk, havuzdan).
SENARYO KOMUTU (her oyuncu, her adım): hedef yön = rakibe; mesafe > YAKLAS_M → v* = 0.35; değilse v* = 0 ve rakibe dönük
  (|açı| < 20°) + ayakta + vuruşta değil + soğuma bitti → vuruş (yumruk/tekme rastgele).
getup_train_par.run_episode ile uyum için GetupEnv defter alanları taklit edilir.
"""
import glob, os
import numpy as np, mujoco, torch
import getup_train as G
import tek_env as T
import tek_vurus_env as TV
import dovus_env as DV
import dovus_mac_env as M
import engebe as EN

OBS_DIM = TV.OBS_DIM + (M.RAKIP_EK if M.RAKIP_GOZ else 0) + (M.SICRAMA_EK if M.SICRAMA else 0)   # 2026-10-25: ek boyutlar sonda
# SIÇRAMA EĞİTİMİ (2026-10-25, kullanıcı hedefi ~0.25 m): öğrenenin vuruş fırsatlarının P_SICRAMA'sı sıçrama (ileri/geri yarı yarıya).
# Ödül: taklit + kök yolu takibi W_SICRAMA·exp(−K_SICRAMA·(yol − ref yol)²) + kesit sonunda ayakta ve yolun ≥ %70'i → W_SICRAMA_BITIS.
P_SICRAMA = float(os.environ.get("P_SICRAMA", "0.3")) if M.SICRAMA else 0.0
W_SICRAMA, K_SICRAMA, W_SICRAMA_BITIS = 1.0, 20.0, 3.0
# SIÇRAMA UZMANI (2026-10-25; ilk karma eğitim 0/16: robot sıçramamayı öğrendi). SICRAMA_UZMAN=1 → rakip uzakta/pasif (gez
# düzeni), öğrenen ayakta ve hazırken HEP sıçrar; her sıçramanın hedefi [SICRAMA_ALT, 1]×SICRAMA_M arasında rastgele (kolaydan zora).
SICRAMA_UZMAN = os.environ.get("SICRAMA_UZMAN", "0") == "1"
SICRAMA_ALT = float(os.environ.get("SICRAMA_ALT", "0.3"))
# Ders 20 (2026-10-26): SICRAMA_ODUL=olay → uçuş/yol ödülü OLAY BAŞINA (inişte bir kez): en uzun TEK uçuş (≤ 0.3 s) ve alınan yolun
# hedefe oranı, yalnız ayakta ve yön DOĞRUYSA; adım başına uçuş ödülü yok (adım başı ödül robotu sekmeye itti).
SICRAMA_ODUL = os.environ.get("SICRAMA_ODUL", "adim")
W_UCUS_OLAY = 3.0
W_UCUS = 0.5                         # sıçrama kesitinde robotun hiçbir parçası yere değmiyorsa (adım değil gerçek sıçrama)
ACT_DIM = T.ACT_DIM
SURE_S = 15.0
BAS_MESAFE, BAS_ACI = (1.0, 2.5), np.radians(60.0)
YAKLAS_M, VURUS_SOGUMA_S, VURUS_ACI = 0.45, 0.6, np.radians(20.0)
W_AYAKTA, W_DONME, W_HIZ, W_DUR = 0.3, 0.6, 0.3, 0.4
# 2026-10-22: bel/omuz kullanınca aktif rakibe karşı ayakta %81 → %72 (ölçüldü); denge turu için ayarlanabilir
W_AYAKTA = float(os.environ.get("W_AYAKTA_DOVUS", W_AYAKTA))
W_ISABET, W_ALINAN, W_NISAN, W_YAKLAS = 1.0, 0.5, 1.0, 0.5
# 2026-10-25 (gardref2: maçta isabet 18 → 23 ama alınan 5.9 → 11.4, düşme 0.9 → 1.6): savunma için ayarlanabilir
W_ALINAN = float(os.environ.get("W_ALINAN", W_ALINAN))
# 2026-10-26 (yeni fizikte yumruk 2.7× hızlı ama isabet/yumruk 1.80 → 0.77): isabete ağırlık için ayarlanabilir
W_ISABET = float(os.environ.get("W_ISABET", W_ISABET)); W_NISAN = float(os.environ.get("W_NISAN", W_NISAN))
# 2026-10-22: bel/omuz-x taklidi (referans ek alanları; govde2'de). Ölçüldü: bu terim yokken yeni eklemler ort 1–2° (kullanılmıyor)
W_EK = float(os.environ.get("W_EK", "0.5"))
EK_EKLEMLER = ("bel_z", "shoulder_x_left", "shoulder_x_right")
# çekirdek: G.MIMIC_EKLEM_K ile benzerlik eklem hiç oynamadan 0.95 çıkıyordu (düz çekirdek, Ders: K4) → 17° fark 0.64, 30° 0.26
K_EK = float(os.environ.get("K_EK", "5.0"))
NISAN_PENCERE_S = 0.25
# YUMRUK HIZI (2026-10-24, kullanıcı: "yumrukları hızlı atamıyorlar"; ölçüldü: en yüksek ~1.8 m/s, insan 8–12): vuruş anı
# penceresinde vuran ucun rakip baş/göğsüne DOĞRU hızı (m/s, HIZ_TAVAN'da doyar) ödüllenir. 0 = kapalı (eski eğitimler aynı).
W_VURUS_HIZ = float(os.environ.get("W_VURUS_HIZ", "0")); HIZ_TAVAN = 5.0
# GARD DURUŞU (2026-10-24, kullanıcı: "bir ayak geride denge yok, dimdik yumruk atıp geri tepmeyle düşüyor"; ölçüldü: insan
# referansı vuruş anında ayaklar ön-arka 0.19 m, gövde öne 20°; robot 0.06 m, 6°). Rakibe GARD_M'den yakınken (dururken ve
# vururken) ayaklar arası ön-arka mesafe GARD_ARA'ya yakınsa W_GARD; vuruş anı penceresinde gövde öne EGIM_DER'e yakınsa W_EGIM.
W_GARD = float(os.environ.get("W_GARD", "0")); W_EGIM = float(os.environ.get("W_EGIM", "0"))
GARD_M, GARD_ARA, EGIM_DER = 1.2, 0.18, 18.0
# GARD BEKLEME (2026-10-27, YAPILACAKLAR 3: "rakibe yakınken hep gardda beklesin, yalnız vururken değil"): ayakta, rakibe GARD_M'den
# yakın, vurmuyor ve gezmiyorken iki kol gard pozunda (dovus_env.GARD_OMUZ/DIRSEK; fizikte PD ile tutuldu) → W_GARD_KOL.
# Kola gelen darbe ×0.25, başa ×2 (dovus_mac_env) → gard alınan hasarı azaltmalı. Ölçüm her zaman: gard_kol = kol hatası < 25° oranı.
# gard1 (K=5): kol hedefe 60–70° uzakken ödül ~0.01 → 20M hiç öğrenmedi (Ders 3 tekrarı, DERSLER 22). Geniş çekirdek: K=0.8 → 70° 0.30, 20° 0.90
# GARD_KILIT=1 (gard3, 2026-10-27): kol ödülüyle öğrenilemedi (ağ kol eylemini +0.5'ten −0.45/−0.84'e taşıyamadı); maçta kuralla gard
# alınanı 14.4 → 5.9 düşürdü ama isabet 38 → 11 (ağ garddan vurmayı hiç görmedi) → eğitimde de kuralla gard (taktik_env.GARD_KILIT ile aynı).
GARD_KILIT = os.environ.get("GARD_KILIT", "0") == "1"
W_GARD_KOL = float(os.environ.get("W_GARD_KOL", "0")); K_GARD_KOL = float(os.environ.get("K_GARD_KOL", "0.8")); GARD_KOL_ESIK = np.radians(25.0)
# GEZİNME bölümleri (2026-10-23). Ölçüldü (v3): v*=0.7 komutunda yürürken 15/45/90° dönüş 0/8 (yön komutu yok sayılıyor),
# gerçek hız ~0.36 m/s; v*=0.35'te dönüyor ama ~18°/s. Sebep: dövüş bölümlerinde yürüme komutu HEP 0.35 ve rakip 1–2.5 m
# yakında → hızlı yürüyüş + yürürken dönme hiç eğitilmemiş. Bölümlerin GEZ_P kadarında rakip uzakta (3–15 m, pasif),
# öğrenen her 2–4 s'de yeni hız (0–1.0) ve yön (±120°) komutu alır; ödül hız + dönme hızı + yön takibi.
GEZ_P = float(os.environ.get("GEZ_P", "0"))
GEZ_HIZLAR, GEZ_HIZ_P = (0.0, 0.35, 0.6, 0.8, 1.0), (0.1, 0.2, 0.25, 0.25, 0.2)
if os.environ.get("GEZ_HIZLAR"):                         # 2026-10-26 gezD: daha hızlı yürüyüş (ör. "0,0.35,0.6,0.8,1.0,1.2"); 0 → %10, kalanı eşit
    GEZ_HIZLAR = tuple(float(x) for x in os.environ["GEZ_HIZLAR"].split(","))
    GEZ_HIZ_P = tuple([0.1] + [0.9 / (len(GEZ_HIZLAR) - 1)] * (len(GEZ_HIZLAR) - 1))
GEZ_OLCUM_N = int(os.environ.get("GEZ_OLCUM_N", "8"))   # ölçüm bölüm sayısı (8 bölümde fark gürültüde kayboluyordu)
GEZ_ARALIK_S, GEZ_YON_MAKS = (2.0, 4.0), np.radians(120.0)
W_GEZ_HIZ, W_GEZ_YON = 0.6, 0.3
PASIF_P = float(os.environ.get("PASIF_P", "0.3"))    # pasif (karşılık vermeyen) rakip oranı
# 2026-10-22 (ölçüldü: ana model sırtüstünden kalkış 0/12): bölümlerin bu kadarında ÖĞRENEN sırtüstü/yüzüstü YERDE başlar
YERDE_P = float(os.environ.get("YERDE_P", "0"))
YERDE_SURE_S = 8.0                     # yerde başlayınca bitişten önce tanınan süre (kalkış ~1.5 s, pay)
# KALKIŞ UZMANI modu (2026-10-22): bölüm GERÇEK DÖVÜŞ DÜŞÜŞÜNDEN başlar (dusus_havuzu.py; sırtüstü/yüzüstü/yan).
# Ölçüldü: TEK_MODELI_v2 temiz sırtüstü pozdan 11/12, dövüş düşüşlerinden 2/9 (yüzüstü 0). Başarı: ayakta 1 s → bölüm biter.
KALKIS_UZMANI = os.environ.get("KALKIS_UZMANI", "0") == "1"
DUSUS_HAVUZU = os.environ.get("DUSUS_HAVUZU", "referanslar/dusus_havuzu_govde2.npz")
W_KALKTI, KALKTI_TUT_S = 5.0, 1.0
YUZUSTU_P = float(os.environ.get("YUZUSTU_P", "0"))
ORNEK_POZ = os.environ.get("ORNEK_POZ", "yuzustu")       # YUZUSTU_P olasılıkla bu sınıftan; ORNEK_POZ != "yuzustu" ise kalanı diğerlerinden
# DÖNME UZMANI modu (2026-10-22): yüzüstü kalkışı RL keşfedemedi (5/22) → iş ikiye bölündü: bu uzman yalnız YÜZÜSTÜNDEN
# SIRTÜSTÜNE döner (sırtüstü kalkış uzmanı 15/15). Ödül: gövde ileri ekseninin dünya-z bileşeni (−1 yüzüstü → +1 sırtüstü)
# artışı + seviye; sırtüstünde 0.3 s → başarı, bölüm biter. Bölüm ≤ 6 s.
DONME_UZMANI = os.environ.get("DONME_UZMANI", "0") == "1"
W_DONME_YUZ, W_DONDU, DONDU_TUT_S, DONME_SURE_S = 3.0, 5.0, 0.3, 6.0
RAKIP_HAVUZ = [f for f in os.environ.get("RAKIP_HAVUZ", "").split(",") if f] or sorted(
    {f for f in ["TEK_MODELI_v2.pth"] + glob.glob("alt_dovus_ep*.pth") + glob.glob("bulut_sonuc/alt_dovus_ep*.pth")
     + glob.glob("alt_dovus2_ep*.pth")                     # v2 kayıtları (govde2 koşusunda: genişletilerek, daha güçlü rakipler)
     if os.path.exists(f)}, key=os.path.basename)
DT = T.DT


_ag_yukle = M.ag_yukle                  # eski robot ağları bu robota genişletilir (govde2)


# ENGEBE (Faz 5, 2026-10-23). Ölçüldü (taktik3+gez, ormanda 96 maç): 14 cm tümsekte düşme artmıyor ama isabet 10.0 → 4.6.
# Bölümlerin ENGEBE_P kadarında zemin tümsekli (tepe-çukur U(3 cm, ENGEBE_MAKS_CM), engebe.py), kalanı düz.
ENGEBE_P = float(os.environ.get("ENGEBE_P", "0"))
ENGEBE_MAKS = float(os.environ.get("ENGEBE_MAKS_CM", "15")) / 100
ENGEBE_YARI, ENGEBE_HARITA = 30.0, 4


class AltDovusEnv:
    def __init__(self, env_id=0, rakip_ck="TEK_MODELI_v2.pth"):
        xml = None; self.engebe_A = 0.0
        if ENGEBE_P > 0:
            xml = os.path.join(os.path.dirname(os.path.abspath(G.XML)), f"_engebe_alt_{os.getpid()}_{env_id}.xml")
            open(xml, "w", encoding="utf-8").write(open(G.XML, encoding="utf-8").read())
            self._engebe = EN.xml_ekle(xml, ENGEBE_MAKS, kutu=((-ENGEBE_YARI,) * 2, (ENGEBE_YARI,) * 2))
            self._engebe_h = [EN.gurultu(self._engebe, 1000 * env_id + k).ravel() for k in range(ENGEBE_HARITA)]
        try:
            self.m_ = M.DovusMacEnv(alt_ck=rakip_ck, env_id=env_id, xml=xml)
        finally:
            if xml:
                os.remove(xml)
        self.model, self.data = self.m_.model, self.m_.data
        self.P = self.m_.P
        self.havuz = [_ag_yukle(ck) for ck in RAKIP_HAVUZ]
        self.havuz_ad = [os.path.basename(c) for c in RAKIP_HAVUZ]
        self.observation_space = type("O", (), {"shape": (OBS_DIM,)})()
        self.p1_actuator_count = ACT_DIM; self.p1_torso_id = self.P[0].torso
        self.max_timesteps = int(round(SURE_S / DT))
        self.assist_force = 0.0; self.assist_steps = 0
        self.mimic_on, self.mimic_sum, self.mimic_released, self.tuck_sum = False, 0.0, False, 0.0
        self.oturma_src = None; self.start_mode = "dovus"; self.r_parca = {}
        m = self.model; B = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        J = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        self._ek_adr = {n: int(m.jnt_qposadr[J(f"p1_{n}")]) for n in EK_EKLEMLER if J(f"p1_{n}") >= 0}
        self._gard_kol_hedef = np.array([DV.GARD_OMUZ, DV.GARD_DIRSEK, DV.GARD_OMUZ, DV.GARD_DIRSEK])
        _qa = list(np.asarray(self.P[0].qa)); self._kol_i = [_qa.index(x) for x in np.asarray(self.P[0].q19)[7:11]]
        self._uclar = [{"yumruk": (B(f"{p}_fist_left"), B(f"{p}_fist_right")),
                        "tekme": (B(f"{p}_foot_left"), B(f"{p}_foot_right"))} for p in ("p1", "p2")]

    def _get_player_position(self, p):
        q = self.P[0] if p == "p1" else self.P[1]; return self.data.xpos[q.bas].copy()

    def mesafe(self):
        return 0.0

    def reset(self, seed=None, options=None):
        if seed is not None: np.random.seed(seed)
        opt = options or {}
        e = self.m_; mujoco.mj_resetData(self.model, self.data)
        mes = np.random.uniform(*BAS_MESAFE); a = np.random.uniform(-np.pi, np.pi)
        x1, y1 = -0.5 * mes * np.cos(a), -0.5 * mes * np.sin(a)
        self.P[0].sifirla(x1, y1, a + np.random.uniform(-BAS_ACI, BAS_ACI))
        self.P[1].sifirla(-x1, -y1, a + np.pi + np.random.uniform(-np.radians(10), np.radians(10)))
        self.yerde_bas = bool(opt.get("yerde", np.random.rand() < YERDE_P))
        self.gez = bool(opt.get("gez", SICRAMA_UZMAN or np.random.rand() < GEZ_P)) and not self.yerde_bas; self._gez_t = 0.0
        if self.gez:                                         # rakip uzakta, rastgele yönde (gözlemde görünür ama ilgisiz)
            ru, ra = np.random.uniform(3.0, 15.0), np.random.uniform(-np.pi, np.pi)
            self.P[1].sifirla(x1 + ru * np.cos(ra), y1 + ru * np.sin(ra), np.random.uniform(-np.pi, np.pi))
        self.donme_bas = bool(opt.get("donme", DONME_UZMANI))
        self.havuz_bas = bool(opt.get("havuz", KALKIS_UZMANI or self.donme_bas)) and not self.gez; self.kalkti = False; self.ayakta_tut = 0
        self.dondu = False; self.sirt_tut = 0; self.__dict__.pop("_iz_onceki", None)
        _sirt = opt.get("sirtustu")
        if self.yerde_bas:                                   # sırtüstü (ya da yüzüstü) yat: gövde ekseni yatay
            p0 = self.P[0]; qa = self.data.qpos[p0.q19[3:7]].copy()
            # y ekseni etrafında −90° = SIRTÜSTÜ (gövde geriye yatar), +90° = yüzüstü; %75 sırtüstü (kalkış bunu öğrendi)
            yat = np.r_[np.cos(np.pi / 4), 0, (-1 if (_sirt if _sirt is not None else np.random.rand() < 0.75) else 1) * np.sin(np.pi / 4), 0]
            q = np.zeros(4); mujoco.mju_mulQuat(q, qa, yat)
            self.data.qpos[p0.q19[3:7]] = q; self.data.qpos[p0.q19[2]] = 0.25
        if self.havuz_bas:                                  # gerçek düşüş durumu: P1'in başlangıç yerine taşı, gürültü ekle
            if not hasattr(self, "_havuz"):
                h = np.load(DUSUS_HAVUZU); self._havuz = (h["qpos"], h["qvel"], h["poz"])
            hq, hv, hp = self._havuz
            if "havuz_i" in opt:
                k = int(opt["havuz_i"])
            elif self.donme_bas or np.random.rand() < YUZUSTU_P:              # 2026-10-22: yüzüstü kalkış zayıf (5/22) → ağırlıklı örnekle
                k = int(np.random.choice(np.where(hp == ("yuzustu" if self.donme_bas else ORNEK_POZ))[0]))
            elif ORNEK_POZ != "yuzustu":                     # v2 havuzu: sınıf dengesi (yuvarlanmış 516 / eski sırt-yan 58)
                k = int(np.random.choice(np.where(hp != ORNEK_POZ)[0]))
            else:
                k = np.random.randint(len(hq))
            p0 = self.P[0]; nq, nv = hq.shape[1], hv.shape[1]
            q = hq[k].copy(); v = hv[k].copy()
            q[0:2] = [x1, y1]; q[7:] += np.random.normal(0, 0.03, nq - 7); v += np.random.normal(0, 0.05, nv)
            self.data.qpos[p0.qs:p0.qs + nq] = q; self.data.qvel[p0.vs:p0.vs + nv] = v
            self.havuz_poz = str(hp[k])
        mujoco.mj_forward(self.model, self.data)
        if ENGEBE_P > 0:
            self._engebe_kur(opt)
        e.t = 0; e.istat = dict(isabet=[0, 0], hasar=[0.0, 0.0], devrilme=[0, 0], ayakta=[0, 0])
        self.rakip_i = int(opt.get("rakip_i", np.random.randint(len(self.havuz))))
        self.rakip_aktif = bool(opt.get("rakip_aktif", (np.random.rand() >= PASIF_P) and not self.havuz_bas)) and not self.gez
        self.timestep = 0; self.vsoguma = [0, 0]; self.dusuk = 0; self.ayakta_adim = 0
        self.isabet, self.isabet_temiz, self.alinan, self.vurus_n = 0, 0, 0, 0
        self.vurus_hiz_top, self.vurus_hiz_n = 0.0, 0
        self.gard_top, self.gard_n, self.egim_top, self.egim_n = 0.0, 0, 0.0, 0
        self.gard_kol_n, self.gard_kol_ok, self.gard_kol_hata = 0, 0, 0.0
        self.sicrama_n, self.sicrama_basari, self.sicrama_yol, self.sicrama_ucus = 0, 0, [], 0.0
        self.gez_n, self.gez_hiz_top, self.gez_hata_top, self.gez_yon_top, self.gez_dus, self._ay_once = 0, 0.0, 0.0, 0.0, 0, True
        self.gez_donus_top, self.gez_donus_n, self._gez_komut_t, self._gez_hizali = 0.0, 0, 0, True
        self.onceki_mes = self._aci_mesafe()[0]
        self._komut(0); self._komut(1)
        e._alt_o = [p.alt_gozlem() for p in self.P]
        return e._alt_o[0], {}

    def _engebe_kur(self, opt):
        """Bu bölümün zemini (opt['engebe'] = genlik m ile zorlanabilir); robotlar tümseğe gömülmesin diye kaldırılır."""
        A = opt.get("engebe", np.random.uniform(0.03, ENGEBE_MAKS) if np.random.rand() < ENGEBE_P else 0.0)
        h = self._engebe_h[np.random.randint(ENGEBE_HARITA)]
        self.model.hfield_data[:] = 0.5 + (A / ENGEBE_MAKS) * h; self.engebe_A = float(A)
        if A > 0:
            for p in self.P:
                x, y = self.data.xpos[p.torso][:2]
                self.data.qpos[p.q19[2]] += max(0.0, EN.zemin_z(self.model, float(x), float(y))) + 0.01
            mujoco.mj_forward(self.model, self.data)

    def _aci_mesafe(self, i=0):
        p, q = self.P[i], self.P[1 - i]; d = self.data
        dx = d.xpos[q.torso][:2] - d.xpos[p.torso][:2]
        return float(np.linalg.norm(dx)), float(np.arctan2(dx[1], dx[0]))

    def _komut(self, i):
        p = self.P[i]
        if i == 0 and SICRAMA_UZMAN:                         # sıçrama uzmanı: hazır olunca sıçra, arada dur
            p.v = 0.0
            if p.vtip is None and p.ayakta_mi() and self.vsoguma[0] <= 0 and p.ayakta_say * DT >= 0.5:
                p.vurus_baslat(M.SICRAMA_TIPLERI[np.random.randint(2)]); self.sicrama_n += 1
                p.s_olcek = float(np.random.uniform(SICRAMA_ALT, 1.0)); self.vsoguma[0] = int(VURUS_SOGUMA_S / DT)
            return
        if i == 0 and self.gez:                              # gezinme: zamanlı rastgele hız/yön komutları
            if self.timestep * DT >= self._gez_t:
                p.v = float(np.random.choice(GEZ_HIZLAR, p=GEZ_HIZ_P))
                p.psi = p.yaw() + np.random.uniform(-GEZ_YON_MAKS, GEZ_YON_MAKS)
                self._gez_t = self.timestep * DT + np.random.uniform(*GEZ_ARALIK_S); self._gez_v = p.v
                self._gez_komut_t, self._gez_hizali = self.timestep, False
            p.v = self._gez_v if p.ayakta_mi() else 0.0
            return
        if p.vtip is not None: return
        if i == 1 and not self.rakip_aktif:
            p.v = 0.0; return                                        # pasif rakip: yerinde durur (yön sabit)
        dist, yon = self._aci_mesafe(i); p.psi = yon
        if not p.ayakta_mi():                                 # yerde: önce kalk (kalkış ortamındaki gibi v*=0)
            p.v = 0.0; return
        if dist > YAKLAS_M:
            p.v = 0.35
        else:
            p.v = 0.0
            if abs(M._sar(yon - p.yaw())) < VURUS_ACI and p.ayakta_mi() and self.vsoguma[i] <= 0:
                if i == 0 and np.random.rand() < P_SICRAMA:
                    p.vurus_baslat(M.SICRAMA_TIPLERI[np.random.randint(2)]); self.sicrama_n += 1
                else:
                    p.vurus_baslat("yumruk" if np.random.rand() < 0.5 else "tekme")
                self.vsoguma[i] = int(VURUS_SOGUMA_S / DT)
                if i == 0: self.vurus_n += 1

    def _nisan(self):
        """Vuruş anı penceresindeyse: vuran uç ↔ rakip baş/göğüs en yakın mesafesi (m), değilse None."""
        p, q = self.P; d = self.data
        if p.vtip not in ("yumruk", "tekme") or p.vgecis > 0: return None
        if abs(p.vk * p.vref["dt"] - p.vref["olay"]) > NISAN_PENCERE_S: return None
        return min(float(np.linalg.norm(d.xpos[u] - d.xpos[h])) for u in self._uclar[0][p.vtip] for h in q.nisan)

    def _gard_egim(self):
        """(ayaklar arası ön-arka mesafe m — gövde yönünde, gövdenin öne eğimi derece)."""
        p = self.P[0]; d = self.data; y = p.yaw(); f = np.array([np.cos(y), np.sin(y)])
        ara = abs(float((d.xpos[p.ayak[0]][:2] - d.xpos[p.ayak[1]][:2]) @ f))
        z = d.xmat[p.torso].reshape(3, 3)[:, 2]                    # gövde yukarı ekseni (dünya)
        return ara, float(np.degrees(np.arctan2(z[0] * f[0] + z[1] * f[1], z[2])))

    def _yere_degiyor(self, p):
        """Oyuncunun herhangi bir parçası zemine (dünya gövdesi) değiyor mu."""
        d, m = self.data, self.model
        for i in range(d.ncon):
            c = d.contact[i]; b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if (b1 == 0 and b2 in p.govde_kume) or (b2 == 0 and b1 in p.govde_kume):
                return True
        return False

    def _vurus_hizi(self):
        """Vuruş anı penceresindeyse: vuran uçların rakip baş/göğsüne doğru en büyük hız bileşeni (m/s, ≥ 0), değilse None."""
        p, q = self.P; d, m = self.data, self.model
        if p.vtip not in ("yumruk", "tekme") or p.vgecis > 0 or abs(p.vk * p.vref["dt"] - p.vref["olay"]) > NISAN_PENCERE_S:
            return None
        v6, en = np.zeros(6), 0.0
        for u in self._uclar[0][p.vtip]:
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, u, v6, 0)
            for h in q.nisan:
                yon = d.xpos[h] - d.xpos[u]; yon /= np.linalg.norm(yon) + 1e-9
                en = max(en, float(v6[3:] @ yon))
        return en

    def step(self, action_p1, hedef=None):
        e, d = self.m_, self.data; p, q = self.P
        self.timestep += 1; self.vsoguma[0] -= 1; self.vsoguma[1] -= 1
        a1 = np.clip(np.asarray(action_p1, np.float64), -1, 1)
        net, mu, sd = self.havuz[self.rakip_i]
        with torch.no_grad():
            a2 = net.eylem((M.kes(torch.as_tensor(e._alt_o[1][None]), mu) - mu) / sd)[0].numpy()[0]
        h = [p.mutlak_hedef(a1) if hedef is None else hedef, q.mutlak_hedef(a2)]
        if GARD_KILIT and hedef is None and not self.gez and p.vtip is None and p.ayakta_mi() and self._aci_mesafe()[0] < GARD_M:
            h[0][self._kol_i] = self._gard_kol_hedef                # kollar kuralla gardda; ağ gard duruşundan vurmayı öğrenir
        verilen = alinan = 0.0
        for _ in range(G.FRAME_SKIP):
            for i, o in enumerate(self.P):
                d.ctrl[o.akt] = np.clip((o.kp * (h[i] - d.qpos[o.qa]) - o.kd * d.qvel[o.va]) / o.g, -1, 1)
            mujoco.mj_step(self.model, d)
            for i, hs, tip, w in e._temaslar():
                if i == 0:
                    verilen += hs; self.isabet += 1; self.isabet_temiz += w >= 1.0
                else:
                    alinan += hs; self.alinan += 1
        for o in self.P:
            o.soguma -= 1; o.durum_ilerlet()
        self._komut(0); self._komut(1)
        e._alt_o = [o.alt_gozlem() for o in self.P]                  # ADIM BAŞINA BİR KEZ (Ders 16)
        # ödül
        ayakta = p.ayakta_mi(); self.ayakta_adim += ayakta; self.dusuk = 0 if ayakta else self.dusuk + 1
        y = p.yaw(); c, s = np.cos(y), np.sin(y); vs = p.vs
        vx, vy = float(d.qvel[vs]), float(d.qvel[vs + 1]); v_ileri = c * vx + s * vy
        dpsi = M._sar(p.psi - y)
        if self.gez:                                         # gezinme ölçümü (2026-10-26): hız oranı/hatası, yön hatası, düşme
            self.gez_dus += int(self._ay_once and not ayakta); self._ay_once = ayakta
            if ayakta and p.v > 0:
                self.gez_n += 1; self.gez_hiz_top += v_ileri / p.v; self.gez_hata_top += abs(v_ileri - p.v)
                self.gez_yon_top += abs(float(dpsi))
            if not self._gez_hizali and abs(float(dpsi)) < np.radians(20):    # dönüş süresi: komuttan yöne (±20°) varışa
                self._gez_hizali = True; self.gez_donus_n += 1
                self.gez_donus_top += (self.timestep - self._gez_komut_t) * DT
        p.wz_ema = 0.95 * p.wz_ema + 0.05 * float(d.qvel[vs + 5])
        r_donme = float(np.exp(-4.0 * (p.wz_ema - np.clip(T.K_DONME * dpsi, -T.W_DONME_MAKS, T.W_DONME_MAKS)) ** 2))
        mes = self._aci_mesafe()[0]; yaklas = (self.onceki_mes - mes) / DT; self.onceki_mes = mes
        ortak = dict(isabet=W_ISABET * verilen, alinan=-W_ALINAN * alinan, eylem=-T.W_EYLEM * float(np.sum(a1 * a1)))
        if not ayakta:
            bas_z = float(d.xpos[p.bas][2]); dik = float(d.xmat[p.torso][8])
            hh = min(bas_z / G.STAND_HEAD_Z, 1.0); prog = 0.3 * max(dik, 0.0) + 0.7 * hh * hh
            self.r_parca = dict(kalkis=-0.6 + 1.2 * prog, **ortak)
        elif p.vtip is not None:
            Tq = len(p.vref["Q"]); ref = p.vref["Q"][min(int(p.vk), Tq - 1)].copy()
            if p.vgecis > 0: ref[7:] *= 1.0 - p.vgecis / p.vgecis_n
            qq = d.qpos[p.q19]; jt = qq[7:][G.MIMIC_JOINTS]; jr = ref[7:][G.MIMIC_JOINTS]
            r_taklit = float(np.mean(np.exp(-G.MIMIC_EKLEM_K * (jt - jr) ** 2)))
            n = self._nisan()
            ek = p.vref.get("EK") or {}
            ortak_ek = [k for k in self._ek_adr if k in ek]
            if ortak_ek:                                      # bel/omuz-x referansı (geçişte dik duruştan harmanlanır)
                w = 1.0 - p.vgecis / p.vgecis_n if p.vgecis > 0 else 1.0; ii = min(int(p.vk), Tq - 1)
                ortak["v_ek"] = W_EK * float(np.mean([np.exp(-K_EK * (d.qpos[self._ek_adr[k]] - w * ek[k][ii]) ** 2)
                                                      for k in ortak_ek]))
            self.r_parca = dict(ayakta=W_AYAKTA, v_taklit=DV.W_TAKLIT * r_taklit, donme=0.5 * W_DONME * r_donme,
                                nisan=0.0 if n is None else W_NISAN * float(np.exp(-10.0 * max(n - 0.1, 0.0) ** 2)), **ortak)
            if p.vtip.startswith("sicrama"):
                if p.vgecis == 0 and not self._yere_degiyor(p):
                    if SICRAMA_ODUL != "olay": self.r_parca["ucus"] = W_UCUS
                    self.sicrama_ucus += DT; p.s_ucus = getattr(p, "s_ucus", 0.0) + DT; p.s_ucus_en = max(getattr(p, "s_ucus_en", 0.0), p.s_ucus)
                else:
                    p.s_ucus = 0.0
                ol = getattr(p, "s_olcek", 1.0)
                yol = p.sicrama_yolu(); ref_yol = ol * float(ref[0] - p.vref["Q"][0, 0]) if p.vgecis == 0 else 0.0
                self.r_parca["sicrama"] = W_SICRAMA * float(np.exp(-K_SICRAMA * (yol - ref_yol) ** 2))
                hedef_yol = ol * float(p.vref["Q"][-1, 0] - p.vref["Q"][0, 0])
                if p.vk >= Tq - 1 and not p.s_odul:          # kesit bitti: ayakta mı, yolun ne kadarı alındı
                    p.s_odul = True; oran = yol / hedef_yol; self.sicrama_yol.append(oran)
                    if SICRAMA_ODUL == "olay":
                        if ayakta and oran > 0:
                            self.r_parca["sicrama_bitis"] = (W_SICRAMA_BITIS * min(oran, 1.0)
                                                             + W_UCUS_OLAY * min(getattr(p, "s_ucus_en", 0.0), 0.3) / 0.3)
                        if ayakta and oran >= 0.7: self.sicrama_basari += 1
                    elif ayakta and oran >= 0.7:
                        self.r_parca["sicrama_bitis"] = W_SICRAMA_BITIS; self.sicrama_basari += 1
            vh = self._vurus_hizi()
            if vh is not None:
                self.vurus_hiz_top += vh; self.vurus_hiz_n += 1
                if W_VURUS_HIZ > 0:
                    self.r_parca["vurus_hiz"] = W_VURUS_HIZ * min(max(vh, 0.0), HIZ_TAVAN) / HIZ_TAVAN
        elif p.v > 0:
            self.r_parca = dict(ayakta=W_AYAKTA, donme=W_DONME * r_donme, hiz=W_HIZ * float(np.exp(-4 * (v_ileri - p.v) ** 2)),
                                yaklas=W_YAKLAS * float(np.clip(yaklas, -1.0, 1.0)), **ortak)
        else:
            self.r_parca = dict(ayakta=W_AYAKTA, donme=W_DONME * r_donme, dur=W_DUR * float(np.exp(-10 * (vx * vx + vy * vy))),
                                **ortak)
        if ayakta and mes < GARD_M and not self.gez:         # gard duruşu / vuruşta öne eğilme (ölçüm her zaman, ödül W_* > 0)
            ara, egim = self._gard_egim()
            self.gard_top += ara; self.gard_n += 1
            if W_GARD > 0 and (p.vtip is not None or p.v == 0):
                self.r_parca["gard"] = W_GARD * float(np.exp(-((ara - GARD_ARA) / 0.07) ** 2))
            if p.vtip is None:
                kol = d.qpos[p.q19][7:11] - self._gard_kol_hedef        # sağ omuz/dirsek, sol omuz/dirsek
                self.gard_kol_n += 1; self.gard_kol_ok += int(np.all(np.abs(kol) < GARD_KOL_ESIK)); self.gard_kol_hata += float(np.mean(np.abs(kol)))
                if W_GARD_KOL > 0:
                    self.r_parca["gard_kol"] = W_GARD_KOL * float(np.mean(np.exp(-K_GARD_KOL * kol * kol)))
            if p.vtip is not None and p.vgecis == 0 and abs(p.vk * p.vref["dt"] - p.vref["olay"]) <= NISAN_PENCERE_S:
                self.egim_top += egim; self.egim_n += 1
                if W_EGIM > 0:
                    self.r_parca["egim"] = W_EGIM * float(np.exp(-((egim - EGIM_DER) / 10.0) ** 2))
        if self.gez and ayakta and p.v > 0 and not SICRAMA_UZMAN:   # gezinme: hız + dönme hızı + yön takibi (yaklaşma yok)
            self.r_parca = dict(ayakta=W_AYAKTA, donme=W_DONME * r_donme, hiz=W_GEZ_HIZ * float(np.exp(-4 * (v_ileri - p.v) ** 2)),
                                yon=W_GEZ_YON * float(np.exp(-T.K_YON * dpsi * dpsi)), **ortak)
        if self.donme_bas:                                  # dönme uzmanı: ödül yalnız yüzüstü → sırtüstü dönme
            iz = float(d.xmat[p.torso][6])
            onceki = getattr(self, "_iz_onceki", iz); self._iz_onceki = iz
            self.r_parca = dict(donme=10.0 * W_DONME_YUZ * (iz - onceki) + 0.3 * iz, eylem=-T.W_EYLEM * float(np.sum(a1 * a1)))
            self.sirt_tut = self.sirt_tut + 1 if iz > 0.6 else 0
            if self.sirt_tut * DT >= DONDU_TUT_S and not self.dondu:
                self.dondu = True; self.r_parca["dondu"] = W_DONDU
            r = float(sum(self.r_parca.values()))
            return e._alt_o[0], r, False, bool(self.dondu or self.timestep * DT >= DONME_SURE_S), {}
        if self.havuz_bas:                                  # kalkış uzmanı: ilk kalkışta başarı ödülü, 1 s ayakta → bitiş
            if ayakta and not self.kalkti:
                self.kalkti = True; self.r_parca["kalkti"] = W_KALKTI
            self.ayakta_tut = self.ayakta_tut + 1 if (ayakta and self.kalkti) else 0
        r = float(sum(self.r_parca.values()))
        sinir = YERDE_SURE_S if ((self.yerde_bas or self.havuz_bas) and self.timestep * DT < YERDE_SURE_S) else 5.0
        terminated = self.dusuk * DT >= sinir
        truncated = self.timestep >= self.max_timesteps or (self.havuz_bas and self.ayakta_tut * DT >= KALKTI_TUT_S)
        return e._alt_o[0], r, terminated, truncated, {}


def alt_dovus_olcum(agent, env, N=8):
    """POZ: N bölüm deterministik, iki mod: PASİF rakip (v1 ile karşılaştırılabilir) ve AKTİF rakip (havuzun ilki =
    TEK_MODELI_v2). isabet (temiz = baş/göğüs) / bölüm, alınan, ayakta oranı, son mesafe, rakibe açı."""
    import torch as _t
    out = []
    for aktif in (False, True):
        isb, tmz, aln, ay, mes, don, vur, vhiz, gard, egim = [], [], [], [], [], [], [], [], [], []
        gk_n, gk_ok, gk_h = 0, 0, 0.0
        sic_n, sic_b, sic_y, sic_u = 0, 0, [], 0.0
        for i in range(N):
            np.random.seed(50_000 + i); _t.manual_seed(50_000 + i)
            # yerde=False: eğitimde YERDE_P açıkken ölçüm de yerde başlıyordu (2026-10-22: ayakta %49 sanıldı, gerçekte %95)
            o, _ = env.reset(options=dict(rakip_i=0, rakip_aktif=aktif, yerde=False)); done = False
            while not done:
                o, _, te, tr, _ = env.step(agent.get_action(o, training=False)[0]); done = te or tr
            dist, yon = env._aci_mesafe()
            isb.append(env.isabet); tmz.append(env.isabet_temiz); aln.append(env.alinan); vur.append(env.vurus_n)
            ay.append(env.ayakta_adim / env.timestep); mes.append(dist); don.append(abs(np.degrees(M._sar(yon - env.P[0].yaw()))))
            if env.vurus_hiz_n:
                vhiz.append(env.vurus_hiz_top / env.vurus_hiz_n)
            if env.gard_n:
                gard.append(env.gard_top / env.gard_n)
            if env.egim_n:
                egim.append(env.egim_top / env.egim_n)
            gk_n += env.gard_kol_n; gk_ok += env.gard_kol_ok; gk_h += env.gard_kol_hata
            sic_n += env.sicrama_n; sic_b += env.sicrama_basari; sic_y += env.sicrama_yol; sic_u += env.sicrama_ucus
        out.append(f"{'AKTIF' if aktif else 'PASIF'}: vurus {np.mean(vur):.1f} isabet {np.mean(isb):.1f} (temiz {np.mean(tmz):.1f}; "
                   f">=1: {sum(x >= 1 for x in isb)}/{N}) alinan {np.mean(aln):.1f} ayakta %{100 * np.mean(ay):.0f} "
                   f"mes {np.median(mes):.2f} aci {np.median(don):.0f} vurus_hizi {np.mean(vhiz) if vhiz else 0:.2f} "
                   f"gard_ara {np.mean(gard) if gard else 0:.2f} egim {np.mean(egim) if egim else 0:.0f} gard_kol %{100 * gk_ok / max(gk_n, 1):.0f} kol_hata {np.degrees(gk_h / max(gk_n, 1)):.0f}"
                   + (f" sicrama {sic_b}/{sic_n} basarili, yol orani ort {np.mean(sic_y) if sic_y else 0:.2f} ucus/sicrama {sic_u / max(sic_n, 1):.2f} s" if P_SICRAMA > 0 else ""))
    if GEZ_P > 0:                                            # GEZİNME (2026-10-26, yürüme 20M eğitimi): N deterministik bölüm
        g_n, g_h, g_e, g_y, g_d, g_a, g_t, g_tn = 0, 0.0, 0.0, 0.0, 0, [], 0.0, 0
        N = GEZ_OLCUM_N
        for i in range(N):
            np.random.seed(60_000 + i); _t.manual_seed(60_000 + i)
            o, _ = env.reset(options=dict(rakip_i=0, gez=True, yerde=False)); done = False
            while not done:
                o, _, te, tr, _ = env.step(agent.get_action(o, training=False)[0]); done = te or tr
            g_n += env.gez_n; g_h += env.gez_hiz_top; g_e += env.gez_hata_top; g_y += env.gez_yon_top; g_d += env.gez_dus
            g_t += env.gez_donus_top; g_tn += env.gez_donus_n
            g_a.append(env.ayakta_adim / env.timestep)
        k = max(g_n, 1)
        out.append(f"GEZ: hiz orani {g_h / k:.2f} hiz hatasi {g_e / k:.2f} m/s yon hatasi {np.degrees(g_y / k):.0f} der "
                   f"dusme {g_d}/{N} bolum ayakta %{100 * np.mean(g_a):.0f} donus {g_t / max(g_tn, 1):.2f} s")
    np.random.seed(None)
    return " || ".join(out)
