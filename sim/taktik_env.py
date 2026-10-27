# -*- coding: utf-8 -*-
"""Taktik ortamı (PLAN: kesintisiz bölgelerde dövüş — Faz 3, 2026-10-22; kullanıcı seçimi B: HİYERARŞİK).

Üç katman:  ÜST  taktik ağ (bu ortamda eğitilir) — her KARAR_ADIM alt adımda (~0.26 s) bir komut seçer
            ORTA gezinme (gezinme.py, kod): kovala / kaç / geri komutlarını A* + 'yay' yol takibiyle yürütür
            ALT  donuk hareket ağı (govde2 dövüş modeli) — iki robot da (dovus_mac_env.DovusMacEnv)
Sahne: robot + bölge zinciri (arena/insa/…_komsuN_fizik.xml; orman → sur gediği → avlu → tapınak).

KOMUTLAR: kovala · yavas_kovala · dur (rakibe dön) · sola · saga · yumruk · tekme · kac (rakipten uzak geçide) · geri
GÖZLEM (47, ego): rakip göreli konum/uzaklık/yön, göreli yönelim ve hız, ayakta/vuruş durumları, canlar, son komut,
  süre, en yakın 2 geçit (göreli konum + uzaklık), 8 yönde engel uzaklığı (ışın, 5 m), bulunulan bölge türü (4).
ÖDÜL (oyuncu başına, DovusMacEnv ile aynı): ± hasar, ± devirme, küçük şekillendirme. Bölge değiştirmeye ZORLANMAZ
  (plan: kaçma/kovalama dövüş ödülünden doğal olarak çıkmalı; ölçülür: 'bolge_degisimi').
kural_komut(): ölçüt rakibi ve taklit (BC) öğretmeni — "uzaksa kovala, menzilde ve dönükse vur, değilse dön".
"""
import glob, json, math, os
import numpy as np, mujoco, torch
import getup_train as G
import dovus_mac_env as M
import gezinme as GZ
import engebe as EN
import dovus_env as DV

# GARD_KILIT="0,1": bu oyuncuların kolları rakibe yakınken (vurmuyorsa) kuralla gardda (_gard_kilitle; 2026-10-27). Boş = kapalı.
GARD_KILIT = tuple(int(x) for x in os.environ.get("GARD_KILIT", "").split(",") if x.strip())
GARD_KILIT_M = float(os.environ.get("GARD_KILIT_M", "1.2"))
# VURUS_FREN_HIZ (2026-10-27): vurus_fren oyuncuları bu yatay hızın (m/s) üstündeyken vuruş komutu "dur"a döner (önce yavaşla).
# Maçlarda taktik oyuncusunun düşmelerinin ~%52'si kendi yumruğunda, öncesinde hız ~1.35 m/s. 0 = kapalı.
VURUS_FREN_HIZ = float(os.environ.get("VURUS_FREN_HIZ", "0"))

KOMUTLAR = ("kovala", "yavas_kovala", "dur", "sola", "saga", "yumruk", "tekme", "kac", "geri")
N_KOMUT = len(KOMUTLAR)
BOLGELER = ("orman", "sur", "avlu", "tapinak")
OBS = 47
MAC_SURE_S = 40.0
UZAK_BAS_P = 0.3                       # bölümlerin bu kadarında rakip KOMŞU bölgede başlar (kovalama gerekir)
MENZIL_M, VURUS_ACI = 0.55, math.radians(20)
ISIN_M = 5.0
ENGEBE_M = float(os.environ.get("ENGEBE_CM", "0")) / 100   # 2026-10-23: orman zemini tümsek genliği (tepe-çukur); 0 = düz
DT = M.DT


class TaktikEnv(M.DovusMacEnv):
    def __init__(self, parca_xml, alt_ck, dunya_klasoru=None, env_id=0):
        xml = GZ.sahne_xml(G.XML, parca_xml)
        try:
            self._engebe = EN.xml_ekle(xml, ENGEBE_M) if ENGEBE_M > 0 else None
            super().__init__(alt_ck=alt_ck, env_id=env_id, xml=xml)
        finally:
            os.remove(xml)
        self.vurus_fren = {0, 1}         # VURUS_FREN_HIZ hangi oyunculara uygulanır (taktik_olc: yalnız taktik oyuncusu)
        m = self.model
        self.harita = GZ.Harita(m)
        site = lambda i: (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, i) or "")
        self.alanlar = sorted([(site(i)[6:-10], m.site_pos[i][:2].copy()) for i in range(m.nsite)
                               if site(i).startswith("mekan_") and site(i).endswith("_acik_alan")], key=lambda t: t[1][1])
        self.gecitler = np.array([m.site_pos[i][:2] for i in range(m.nsite) if "_kapi_" in site(i)])
        tur = {}
        for f in glob.glob(os.path.join(dunya_klasoru or "", "odalar", "*.json")):
            o = json.load(open(f, encoding="utf-8")); tur[o["id"]] = o.get("_bolge")
        self.alan_turu = [tur.get(oid) for oid, _ in self.alanlar]
        if self._engebe:                                  # tümsekler yalnız ORMAN bölgesinde (sınır geomlarının kutusu)
            mujoco.mj_forward(m, self.data); kutular = []
            for oid, t in tur.items():
                xy = np.array([self.data.geom_xpos[g][:2] for g in range(m.ngeom)
                               if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith(f"mekan_{oid}_sinir")])
                if t == "orman" and len(xy):
                    kutular.append((*xy.min(0), *xy.max(0)))
            EN.doldur(m, self._engebe, kutular, tohum=env_id); self.engebe_kutular = kutular
        self.yt = [None, None]
        self.kalkis = None; self._kalkista = [False, False]; self._kalkis_say = [0, 0]   # beceri seçici (kalkis_yukle)
        self.gez = None; self._gezde = [False, False]                                     # uzakta gezinme ağı (gez_yukle)
        self._isin_yon = [np.array([math.cos(a), math.sin(a), 0.0]) for a in np.arange(8) * math.pi / 4]

    def donme_yukle(self, ck):
        """Yüzüstü yatarken DÖNME uzmanı sürer (yüzüstü → sırtüstü), sonra kalkış uzmanı (2026-10-22)."""
        if ck.endswith(".npz"):                             # açık döngü yuvarlanma (beceri_secici.DonmeHareketi), oyuncu başına
            from beceri_secici import DonmeHareketi
            self.donme, self._donme_h = None, [DonmeHareketi(ck, DT), DonmeHareketi(ck, DT)]
        else:
            self.donme = M.ag_yukle(ck)

    def kalkis_yukle(self, ck, donus_s=0.3):
        """Beceri seçici (2026-10-22, beceri_secici.py ile aynı kural): robot düşünce bu KALKIŞ UZMANI sürer,
        kalkıp donus_s ayakta kalınca dövüş ağına döner."""
        self.kalkis = M.ag_yukle(ck); self._donus_n = int(round(donus_s / DT))

    def ikinci_alt_yukle(self, ck, ref_ayar=None, gecis_s=None, ref_ayar0=None, gecis_s0=None):
        """2026-10-25 kafa kafaya maç: MAVİ (P2) başka bir hareket ağı + kendi referans ayarı (VURUS_ONCE_S, VURUS_SONRA_S,
        GARD_REF) ve geçiş süresiyle; KIRMIZI'nın ayarı da ref_ayar0/gecis_s0 ile verilebilir. Kalkış/gezinme ortak."""
        self._net2 = M.ag_yukle(ck)
        self.P[1].ref_ayar, self.P[1].gecis_s = ref_ayar, gecis_s
        self.P[0].ref_ayar, self.P[0].gecis_s = ref_ayar0, gecis_s0

    def gez_yukle(self, ck, uzak_m=2.5, yakin_m=1.8, oyuncular=(0, 1)):
        """2026-10-23: rakip UZAKTAYKEN gezinme ağı (gez_epson4M: gerçek hız 0.59 m/s, dönüş daha iyi), yakında dövüş ağı
        (v3: savunması daha iyi; gez ağıyla alınan darbe 1.3 → 2.5). Eşikler histerezisli: > uzak_m gez, < yakin_m dövüş."""
        self.gez = M.ag_yukle(ck); self._gez_esik = (uzak_m, yakin_m); self.gez_oyuncular = set(oyuncular)

    def _gez_sec(self, a, X):
        if self.gez is None:
            return a
        d = float(np.linalg.norm(self._xy(0) - self._xy(1))); uz, yk = self._gez_esik; sec = []
        for i, p in enumerate(self.P):
            self._gezde[i] = i in self.gez_oyuncular and p.vtip is None and d > (yk if self._gezde[i] else uz)
            if self._gezde[i]:
                sec.append(i); self.istat["gez"][i] += 1
        if sec:
            net, mu, sd = self.gez
            with torch.no_grad():
                ag = net.eylem((M.kes(X, mu) - mu) / sd)[0].numpy()
            for i in sec:
                a[i] = ag[i]
        return a

    def _kalkis_sec(self, a):
        if self.kalkis is None:
            return a
        net, mu, sd = self.kalkis; ak = None
        for i, p in enumerate(self.P):
            if not p.ayakta_mi():
                self._kalkista[i], self._kalkis_say[i] = True, 0
            elif self._kalkista[i]:
                self._kalkis_say[i] += 1
                if self._kalkis_say[i] >= self._donus_n:
                    self._kalkista[i] = False
            yuzustu = not p.ayakta_mi() and float(self.data.xmat[p.torso][6]) < -0.3
            dh = getattr(self, "_donme_h", None)
            if dh is not None:
                if self._kalkista[i] and yuzustu:
                    a[i] = dh[i].get_action(None)[0]; continue
                dh[i].sifirla()
            if self._kalkista[i] and getattr(self, "donme", None) is not None and yuzustu:
                net2, mu2, sd2 = self.donme
                with torch.no_grad():
                    a[i] = net2.eylem((M.kes(torch.as_tensor(self._alt_o[i][None]), mu2) - mu2) / sd2)[0].numpy()[0]
                continue
            if self._kalkista[i]:
                if ak is None:
                    with torch.no_grad():
                        ak = net.eylem((M.kes(torch.as_tensor(np.stack(self._alt_o)), mu) - mu) / sd)[0].numpy()
                a[i] = ak[i]
        return a

    # ───────────────────────────────────────────── yardımcılar
    def _xy(self, i):
        return self.data.xpos[self.P[i].torso][:2].copy()

    # 2026-10-24 (canavar görevi): "rakip" tek yerden — maçta öbür robot, canavar_env'de canavar
    def rakip_xy(self, i):
        return self._xy(1 - i)

    def rakip_bilgi(self, i):
        """(yaw, ayakta, vuruş [yumruk, tekme, ilerleme], can 0..100, dünya hızı xy)"""
        q = self.P[1 - i]
        vur = ([1.0, 0.0] if q.vtip == "yumruk" else [0.0, 1.0] if q.vtip == "tekme" else [0.0, 0.0]) + \
              [(q.vk / len(q.vref["Q"])) if q.vtip else 0.0]
        return q.yaw(), float(q.ayakta_mi()), vur, q.hp, self.data.qvel[q.vs:q.vs + 2].copy()

    def bolge(self, i):
        p = self._xy(i)
        return int(np.argmin([np.linalg.norm(p - c) for _, c in self.alanlar]))

    def _yerlestir(self, i, xy, yaw):
        self.P[i].sifirla(float(xy[0]), float(xy[1]), float(yaw))
        if self._engebe:                                  # tümseğe gömülerek başlamasın
            self.data.qpos[self.P[i].q19[2]] += max(0.0, EN.zemin_z(self.model, float(xy[0]), float(xy[1]))) + 0.01

    # ───────────────────────────────────────────── bölüm
    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        mujoco.mj_resetData(self.model, self.data)
        b = np.random.randint(len(self.alanlar)); c = self.alanlar[b][1]
        a = np.random.uniform(-np.pi, np.pi); mes = np.random.uniform(1.2, 2.6)
        p1 = c - 0.5 * mes * np.array([math.cos(a), math.sin(a)])
        if np.random.rand() < UZAK_BAS_P and len(self.alanlar) > 1:          # rakip komşu bölgede
            b2 = b + (1 if b + 1 < len(self.alanlar) else -1)
            p2 = self.alanlar[b2][1] + np.random.uniform(-1, 1, 2)
        else:
            p2 = c + 0.5 * mes * np.array([math.cos(a), math.sin(a)])
        y1 = math.atan2(p2[1] - p1[1], p2[0] - p1[0]); y2 = y1 + math.pi
        self._yerlestir(0, p1, y1 + np.random.uniform(-0.35, 0.35)); self._yerlestir(1, p2, y2 + np.random.uniform(-0.35, 0.35))
        mujoco.mj_forward(self.model, self.data)
        self.t = 0
        self.istat = dict(isabet=[0, 0], hasar=[0.0, 0.0], devrilme=[0, 0], ayakta=[0, 0], bolge_degisimi=[0, 0],
                          komut=np.zeros((2, N_KOMUT), int), gez=[0, 0])
        self._gezde = [False, False]
        self._bolge_son = [self.bolge(0), self.bolge(1)]
        self.yt = [None, None]
        self._alt_o = [p.alt_gozlem() for p in self.P]
        return self.ust_gozlem(0), self.ust_gozlem(1)

    def _komut_uygula(self, i, k):
        p = self.P[i]; p.son_komut = k; self.istat["komut"][i, k] += 1
        if p.vtip is not None:
            return
        ad = KOMUTLAR[k]; ben, o = self._xy(i), self.rakip_xy(i)
        if (ad in ("yumruk", "tekme") and VURUS_FREN_HIZ > 0 and i in self.vurus_fren
                and np.linalg.norm(self.data.qvel[p.vs:p.vs + 2]) > VURUS_FREN_HIZ):
            ad = "dur"; self.istat.setdefault("fren", [0, 0])[i] += 1
        self.yt[i] = None
        if ad in ("kovala", "yavas_kovala"):
            self.yt[i] = (GZ.YolTakip(self.harita, tuple(o)), 0.7 if ad == "kovala" else 0.35, "rakip")
        elif ad == "kac":
            # rakipten uzak, bana yakın geçit: (rakibe uzaklık − bana uzaklık) en büyük
            g = self.gecitler[np.argmax(np.linalg.norm(self.gecitler - o, axis=1) - np.linalg.norm(self.gecitler - ben, axis=1))]
            self.yt[i] = (GZ.YolTakip(self.harita, tuple(g)), 0.7, None)
        elif ad == "geri":
            u = ben - o; u = u / (np.linalg.norm(u) + 1e-6)
            self.yt[i] = (GZ.YolTakip(self.harita, tuple(ben + 3.0 * u)), 0.35, None)
        elif ad == "dur":
            p.v = 0.0; p.psi = math.atan2(o[1] - ben[1], o[0] - ben[0])
        elif ad in ("sola", "saga"):
            p.v = 0.0; p.psi = p.yaw() + (math.pi / 4 if ad == "sola" else -math.pi / 4)
        elif p.ayakta_mi():
            p.vurus_baslat(ad)

    def _yol_guncelle(self, i):
        if self.yt[i] is None or self.P[i].vtip is not None:
            return
        yt, vmaks, izle = self.yt[i]
        ben = self._xy(i)
        if izle == "rakip":                                   # hedef rakip: yer değiştirdikçe güncelle
            o = tuple(self.rakip_xy(i))
            if math.hypot(o[0] - yt.hedef[0], o[1] - yt.hedef[1]) > 1.0:
                yt.hedef, yt.W = o, None
            if yt.W is None and self.harita.gorur(tuple(ben), o):
                yt.W, yt.k = [o], 0                           # görüş hattı açık: A* gerekmez
        psi, v = yt.komut(tuple(ben), self.P[i].yaw())
        self.P[i].psi, self.P[i].v = psi, min(v, vmaks) if not yt.vardi(ben) else 0.0

    def _gard_kilitle(self, i, h):
        """GARD_KILIT (2026-10-27): ayakta, rakibe GARD_KILIT_M'den yakın ve vurmuyorsa iki kolun PD hedefi gard pozu
        (dovus_env.GARD_OMUZ/DIRSEK). Öğrenilemedi (gard1/gard2: ağ kol eylemini +0.5'ten −0.45/−0.84'e taşıyamadı, sigma 0.08);
        zorlayınca kollar −66/−108°, robot 600/600 adım ayakta, vurmaya devam (ölçüldü)."""
        p, q = self.P[i], self.P[1 - i]
        if p.vtip is not None or not p.ayakta_mi():
            return
        if np.linalg.norm(self.data.xpos[p.torso][:2] - self.data.xpos[q.torso][:2]) > GARD_KILIT_M:
            return
        if not hasattr(p, "_kol_i"):
            qa = list(np.asarray(p.qa)); p._kol_i = [qa.index(x) for x in np.asarray(p.q19)[7:11]]
        h[p._kol_i] = (DV.GARD_OMUZ, DV.GARD_DIRSEK, DV.GARD_OMUZ, DV.GARD_DIRSEK)

    def adim(self, k1, k2):
        """DovusMacEnv.adim ile aynı fizik/ödül; ek: gezinme komutları her alt adımda güncellenir, bölge değişimi sayılır."""
        self._komut_uygula(0, k1); self._komut_uygula(1, k2)
        r = [0.0, 0.0]
        for _ in range(M.KARAR_ADIM):
            for i in (0, 1):
                self._yol_guncelle(i)
            self._alt_o = [p.alt_gozlem() for p in self.P]
            X = torch.as_tensor(np.stack(self._alt_o))
            with torch.no_grad():
                a = self._net.eylem((M.kes(X, self._mu) - self._mu) / self._sd)[0].numpy()
                if getattr(self, "_net2", None) is not None:      # kafa kafaya: mavi kendi hareket ağıyla
                    n2, mu2, sd2 = self._net2
                    a[1] = n2.eylem((M.kes(X[1:2], mu2) - mu2) / sd2)[0].numpy()[0]
            a = self._gez_sec(a, X)
            a = self._kalkis_sec(a)
            hed = [self.P[i].mutlak_hedef(a[i]) for i in (0, 1)]
            d = self.data
            for i in GARD_KILIT:
                self._gard_kilitle(i, hed[i])
            for _s in range(G.FRAME_SKIP):
                for i, p in enumerate(self.P):
                    d.ctrl[p.akt] = np.clip((p.kp * (hed[i] - d.qpos[p.qa]) - p.kd * d.qvel[p.va]) / p.g, -1, 1)
                mujoco.mj_step(self.model, d)
                for i, h, tip, _w in self._temaslar():
                    s = self.P[1 - i]; s.hp = max(0.0, s.hp - h)
                    r[i] += M.W_HASAR * h; r[1 - i] -= M.W_HASAR * h
                    self.istat["isabet"][i] += 1; self.istat["hasar"][i] += h
            self.t += 1
            for i, p in enumerate(self.P):
                p.soguma -= 1; p.durum_ilerlet()
                ay = p.ayakta_mi(); self.istat["ayakta"][i] += ay
                if p.ayakta_onceki and p.dustu_mu():
                    p.devrilme += 1; self.istat["devrilme"][i] += 1; r[i] -= M.W_DEVIR; r[1 - i] += M.W_DEVIR
                p.ayakta_onceki = not p.dustu_mu()
                p.yerde = 0 if ay else p.yerde + 1
        for i in (0, 1):
            b = self.bolge(i)
            if b != self._bolge_son[i]:
                self.istat["bolge_degisimi"][i] += 1; self._bolge_son[i] = b
            p, q = self.P[i], self.P[1 - i]
            dx = self._xy(1 - i) - self._xy(i); dist = float(np.linalg.norm(dx))
            don = math.cos(M._sar(math.atan2(dx[1], dx[0]) - p.yaw()))
            r[i] += M.W_DONUK * don + M.W_MESAFE * float(np.exp(-4 * (dist - 0.6) ** 2)) - M.W_PASIF
        bitti = (self.t * DT >= MAC_SURE_S or any(p.devrilme >= M.DEVRILME_MAKS for p in self.P)
                 or any(p.yerde * DT >= M.YERDE_MAKS_S for p in self.P))
        return self.ust_gozlem(0), self.ust_gozlem(1), r[0], r[1], bitti, self.istat

    def ust_gozlem(self, i):
        p = self.P[i]; d = self.data
        y = p.yaw(); c, s = math.cos(y), math.sin(y)
        ego = lambda v: np.array([c * v[0] + s * v[1], -s * v[0] + c * v[1]])
        ben = self._xy(i); dx = self.rakip_xy(i) - ben; rel = ego(dx); dist = float(np.linalg.norm(dx))
        q_yaw, q_ayakta, q_vur, q_hp, q_v = self.rakip_bilgi(i)
        aci = math.atan2(rel[1], rel[0]); dyaw = M._sar(q_yaw - y)
        rv = ego(q_v - d.qvel[p.vs:p.vs + 2])
        vur = lambda o: ([1.0, 0.0] if o.vtip == "yumruk" else [0.0, 1.0] if o.vtip == "tekme" else [0.0, 0.0]) + \
                        [(o.vk / len(o.vref["Q"])) if o.vtip else 0.0]
        sk = np.zeros(N_KOMUT); sk[p.son_komut if p.son_komut < N_KOMUT else 2] = 1
        gd = np.linalg.norm(self.gecitler - ben, axis=1); en = np.argsort(gd)[:2]
        gec = np.concatenate([np.r_[ego(self.gecitler[j] - ben) / 10.0, gd[j] / 10.0] for j in en]) if len(en) == 2 \
            else np.zeros(6)
        isin = np.full(8, 1.0); gid = np.zeros(1, np.int32)
        bas = np.r_[ben, 0.6]
        for k, u in enumerate(self._isin_yon):
            w = np.r_[c * u[0] - s * u[1], s * u[0] + c * u[1], 0.0]          # ego yönü dünyaya
            ds = mujoco.mj_ray(self.model, d, bas + 0.35 * w, w, None, 1, p.torso, gid)
            if 0 <= ds < ISIN_M:
                isin[k] = ds / ISIN_M
        bt = np.zeros(len(BOLGELER)); tur = self.alan_turu[self.bolge(i)] if self.alan_turu else None
        if tur in BOLGELER:
            bt[BOLGELER.index(tur)] = 1
        return np.r_[rel / 5.0, dist / 5.0, math.sin(aci), math.cos(aci), math.sin(dyaw), math.cos(dyaw), rv,
                     float(p.ayakta_mi()), q_ayakta, vur(p), q_vur, p.hp / 100.0, q_hp / 100.0, sk,
                     self.t * DT / MAC_SURE_S, gec, isin, bt].astype(np.float32)


def kural_komut(env: TaktikEnv, i: int, sira=[0]) -> int:
    """Ölçüt rakibi / BC öğretmeni: uzaksa kovala; menzilde ve rakibe dönükse sırayla yumruk/tekme; değilse rakibe dön."""
    p = env.P[i]; ben, o = env._xy(i), env.rakip_xy(i)
    if not p.ayakta_mi() or p.vtip is not None:
        return KOMUTLAR.index("dur")
    dist = float(np.linalg.norm(o - ben))
    if dist > MENZIL_M:
        return KOMUTLAR.index("kovala")
    hata = abs(M._sar(math.atan2(o[1] - ben[1], o[0] - ben[0]) - p.yaw()))
    if hata < VURUS_ACI:
        sira[0] += 1
        return KOMUTLAR.index("yumruk" if sira[0] % 2 else "tekme")
    return KOMUTLAR.index("dur")
