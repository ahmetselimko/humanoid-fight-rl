# -*- coding: utf-8 -*-
"""İki robotlu DÖVÜŞ ortamı (2026-10-21, onaylı plan F1): hiyerarşik — üst (taktik) komut, alt TEK_MODELI_v2 hareket.

Alt ağ (donuk, 109 gözlem, mutlak PD) HER İKİ robotu da sürer; her robot kendi ego çerçevesinde TekEnv/TekVurusEnv ile aynı
anlamda gözlem alır (rakip boyutları kilitli — alt ağ değişmesin). Komut durumları (v*, hedef yön, yürüme fazı, vuruş) oyuncu
başına tutulur; TekEnv.step ve TekVurusEnv.step'teki durum makineleri burada oyuncu başına yeniden kurulmuştur.
Üst komut (KARAR_ADIM = 16 alt adım ≈ 0.26 s'de bir): 0 ilerle · 1 yavaş · 2 dur · 3 sola dön · 4 sağa dön · 5 yumruk · 6 tekme.
  ilerle/yavaş/dur: hedef yön = rakibe doğru; dön: yön ±45°; vuruş: robot ayakta ve vuruşta değilse başlar.
Hasar: saldıranın yumruk/ayak gövdesi ↔ savunanın baş/gövde teması, temas kuvveti > HASAR_F_MIN; vuruş başına en fazla bir
  isabet (soğuma). hasar = min(F/80, 8) × (baş 2 / gövde 1) (selfplay_training.ImprovedRewardSystem fikri).
Üst ödül (oyuncu başına, sıfır toplamlıya yakın): +verilen − alınan hasar, ±devirme, küçük şekillendirme (rakibe dönüklük,
  uygun mesafe), pasiflik cezası. Bölüm: 30 s / 3 devrilme / 8 s yerde.
"""
import os
import numpy as np, mujoco, torch
import getup_train as G
import tek_env as T
import tek_vurus_env as TV
import dovus_env as DV

KOMUTLAR = ("ilerle", "yavas", "dur", "sola", "saga", "yumruk", "tekme")
KARAR_ADIM = 16
MAC_SURE_S, YERDE_MAKS_S, DEVRILME_MAKS = 30.0, 8.0, 3
HASAR_F_MIN, HASAR_SOGUMA_S = 40.0, 0.3
W_HASAR, W_DEVIR, W_DONUK, W_MESAFE, W_PASIF = 0.1, 2.0, 0.01, 0.01, 0.002
DT = T.DT
UST_OBS = 27
BAS_MESAFE, BAS_YON_MAKS = (0.55, 0.8), np.radians(10.0)   # ilk sürüm: vuruş menzilinde, karşılıklı (dönme zayıf)


def _sar(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# RAKİBİ GÖRME (2026-10-25, kullanıcı: "2'yi en baştan yapmamamız hataydı"; gardref3 teşhisi: rakip boyutları kilitli →
# öğrenen gelen yumruğu göremiyor, darbe cezası artınca savunma değil yalnız denge öğrenebildi). RAKIP_GOZ=1 → alt gözlemin
# SONUNA RAKIP_EK boyut eklenir (Oyuncu.rakip_ozellik); ağ ag_genislet ile sıfır ağırlıkla genişletilir. Eski ağlar (kalkış,
# gezinme, rakip havuzu) gözlemin yalnız ilk OBS_DIM boyutunu alır (kes()).
RAKIP_EK = 12
RAKIP_GOZ = os.environ.get("RAKIP_GOZ", "0") == "1"
# SIÇRAMA BECERİSİ (2026-10-25): SICRAMA=1 → alt gözlemin en SONUNA [sıçra_ileri, sıçra_geri] (rakip boyutlarından sonra).
SICRAMA_EK = 2
SICRAMA = os.environ.get("SICRAMA", "0") == "1"
SICRAMA_TIPLERI = ("sicrama_ileri", "sicrama_geri")


def vurus_kodu(tip):
    """Vuruş tek-sıcak kodu: yumruk (1, 0), tekme (0, 1), diğer (sıçrama / yok) (0, 0)."""
    return [1.0, 0.0] if tip == "yumruk" else [0.0, 1.0] if tip == "tekme" else [0.0, 0.0]


def kes(X, mu):
    """Ağın girdi boyutuna kes (yeni boyutlar sonda): eski 119'luk ağlar rakip-gözlemli gözlemle de çalışır."""
    return X[..., :len(mu)]


class Oyuncu:
    """Bir robotun adresleri + komut/yürüme/vuruş durumu."""
    def __init__(self, env, p):
        m = env.model; self.p = p; self.env = env
        jid = lambda a: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{p}_{a}")
        bid = lambda a: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{p}_{a}")
        kok = jid("root"); qs, vs = int(m.jnt_qposadr[kok]), int(m.jnt_dofadr[kok]); self.qs, self.vs = qs, vs
        self.q19 = np.r_[np.arange(qs, qs + 7), [m.jnt_qposadr[jid(a)] for a in G.ESKI_EKLEMLER]]
        self.v18 = np.r_[np.arange(vs, vs + 6), [m.jnt_dofadr[jid(a)] for a in G.ESKI_EKLEMLER]]
        p1_akt = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)
                  if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i).startswith("p1_")]
        self.akt = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a.replace("p1_", f"{p}_", 1)) for a in p1_akt])
        aj = [m.actuator_trnid[i, 0] for i in self.akt]
        self.qa = np.array([m.jnt_qposadr[j] for j in aj]); self.va = np.array([m.jnt_dofadr[j] for j in aj])
        self.g = m.actuator_gear[self.akt, 0].copy()
        self.kp, self.kd = G.pd_kazanc([mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in aj])
        self.lo, self.hi = m.jnt_range[aj, 0].copy(), m.jnt_range[aj, 1].copy()
        self.orta, self.yari = 0.5 * (self.lo + self.hi), 0.5 * (self.hi - self.lo)
        self.govdeler = [b for b in range(m.nbody) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith(p)]
        self.kutle = m.body_mass[self.govdeler]
        self.govde_kume = set(self.govdeler)
        self.torso, self.bas = bid("torso"), bid("head")
        self.ayak = (bid("foot_left"), bid("foot_right"))
        self.vuran = {bid("fist_left"), bid("fist_right"), bid("foot_left"), bid("foot_right")}
        # 2026-10-21 (ölçüldü): 6M modelin yumrukları rakibin önde duran KOLLARINA çarpıyordu (591 N), baş/göğüs hiç → isabet
        # ödülü hiç gelmedi. Kısmi isabet: gard (kol/yumruk) ×0.25, uyluk (alçak tekme) ×0.5; baş ×2, göğüs ×1 aynı.
        gogus = bid("ust_govde") if bid("ust_govde") >= 0 else bid("torso")   # govde2: göğüs kapsülü üst gövdede
        self.hedef_govde = {bid("head"): 2.0, gogus: 1.0, bid("thigh_left"): 0.5, bid("thigh_right"): 0.5,
                            **{bid(n): 0.25 for n in ("left_upper_arm", "right_upper_arm", "forearm_left", "forearm_right",
                                                      "fist_left", "fist_right")}}
        self.nisan = (bid("head"), gogus)          # nişan ödülü hedefleri (temiz vuruş)
        self.vuran_tip = {bid("fist_left"): "yumruk", bid("fist_right"): "yumruk",
                          bid("foot_left"): "tekme", bid("foot_right"): "tekme"}

    # ── durum ──
    def sifirla(self, x, y, yaw):
        e, d = self.env, self.env.data
        q = np.zeros(19); q[0], q[1], q[2] = x, y, 0.62 + G.GOVDE_EK
        q[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        d.qpos[self.q19] = q; d.qvel[self.v18] = 0.0
        self.v, self.psi, self.faz_u, self.yuruyor, self.ayakta_say = 0.0, yaw, 0.0, False, 0
        self.dongu_i = np.random.randint(len(e._D)); self.wz_ema = 0.0; self.prev_com = None
        self.vtip, self.vref, self.vgecis, self.vk, self.vtut, self.vbas = None, None, 0, 0.0, 0, 0
        self.hp, self.devrilme, self.yerde, self.ayakta_onceki = 100.0, 0, 0, True
        self.soguma, self.son_komut, self.son_isabet_t = 0, 2, -9.0

    def yaw(self):
        xm = self.env.data.xmat[self.torso]; return float(np.arctan2(xm[3], xm[0]))

    def rakip_ozellik(self):
        """RAKIP_EK = 12, hepsi KENDİ yönüme göre: rakip gövde göreli (x, y) · rakip bakış farkı (sin, cos) · rakip vuruş
        tipi (yumruk, tekme) · rakip vuruş fazı (sin πu, cos πu; vuruş yoksa 0, 1) · rakibin BAŞIMA en yakın vuran ucu
        (yumruk/ayak) göreli (x, y, z) · o ucun başıma yaklaşma hızı (m/s / 5). [−3, 3] kırpılır."""
        r = getattr(self, "rakip", None)
        if r is None:
            return np.zeros(RAKIP_EK, np.float32)
        d, m = self.env.data, self.env.model
        y = self.yaw(); c, s = np.cos(y), np.sin(y)
        don = lambda v: np.array([c * v[0] + s * v[1], -s * v[0] + c * v[1], v[2]])
        g = don(d.xpos[r.torso] - d.xpos[self.torso])
        dy = (r.yaw() - y + np.pi) % (2 * np.pi) - np.pi
        if r.vtip is None:
            vt, fz = [0.0, 0.0], [0.0, 1.0]
        else:
            u = (-TV.GECIS_U * r.vgecis / r.vgecis_n) if r.vgecis > 0 else r.vk / len(r.vref["Q"])
            vt = vurus_kodu(r.vtip); fz = [np.sin(np.pi * u), np.cos(np.pi * u)]
        bas = d.xpos[self.bas]; uc = min(r.vuran, key=lambda b: float(np.sum((d.xpos[b] - bas) ** 2)))
        fark = d.xpos[uc] - bas; uz = float(np.linalg.norm(fark)) + 1e-6
        v_uc, v_ben = np.zeros(6), np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, uc, v_uc, 0)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, self.bas, v_ben, 0)
        yaklas = -float((v_uc[3:] - v_ben[3:]) @ fark) / uz / 5.0
        o = np.r_[g[:2], np.sin(dy), np.cos(dy), vt, fz, don(fark), yaklas]
        return np.clip(o, -3, 3).astype(np.float32)

    def ayakta_mi(self):
        d = self.env.data
        return d.xpos[self.bas][2] > G.STAND_Z and d.xmat[self.torso][8] > G.STAND_TORSO

    def dustu_mu(self):
        d = self.env.data
        return d.xpos[self.bas][2] < T.Y.DUSME_Z or d.xmat[self.torso][8] < 0.5

    def com_ozellik(self):
        d = self.env.data
        com = (d.xipos[self.govdeler] * self.kutle[:, None]).sum(0) / self.kutle.sum()
        feet = 0.5 * (d.xpos[self.ayak[0]] + d.xpos[self.ayak[1]])
        vel = np.zeros(3) if self.prev_com is None else (com - self.prev_com) / DT
        self.prev_com = com.copy()
        y = self.yaw(); c, s = np.cos(y), np.sin(y); e_ = com - feet
        return np.array([c * e_[0] + s * e_[1], -s * e_[0] + c * e_[1], e_[2], c * vel[0] + s * vel[1], -s * vel[0] + c * vel[1]])

    def alt_gozlem(self):
        """TEK_MODELI_v2 gözlemi (109) — TekVurusEnv.get_obs_both ile aynı düzen. ADIM BAŞINA BİR KEZ (com hızı, Ders 16)."""
        e = self.env
        o95 = e._get_obs(self.p)
        o100 = T.gozlem_kilitle(np.r_[o95, self.com_ozellik()])
        faz = [np.sin(2 * np.pi * self.faz_u), np.cos(2 * np.pi * self.faz_u)] if self.yuruyor else [0.0, 1.0]
        dpsi = _sar(self.psi - self.yaw())
        if self.vtip is None:
            vg = [0.0, 0.0, 0.0, 1.0]
        else:
            u = (-TV.GECIS_U * self.vgecis / self.vgecis_n) if self.vgecis > 0 else self.vk / len(self.vref["Q"])
            vg = vurus_kodu(self.vtip) + [np.sin(np.pi * u), np.cos(np.pi * u)]
        o = np.r_[o100, faz, [self.v / T.V_OLCEK, np.sin(dpsi), np.cos(dpsi)], vg].astype(np.float32)
        if RAKIP_GOZ:
            o = np.r_[o, self.rakip_ozellik()]
        if SICRAMA:
            o = np.r_[o, [float(self.vtip == t) for t in SICRAMA_TIPLERI]]
        return o.astype(np.float32)

    def mutlak_hedef(self, a):
        return np.clip(self.orta + np.clip(a, -1, 1) * G.PD_ASIM * self.yari, self.lo, self.hi)

    def vurus_baslat(self, tip):
        # 2026-10-25: oyuncuya özel referans ayarı (kafa kafaya maç: eski ve yeni hareket ağı farklı referansla eğitildi)
        R = DV.dovus_refleri(tip, getattr(self, "ref_ayar", None)); self.vtip = tip; self.vref = R[np.random.randint(len(R))]
        self.vgecis_n = int(round((getattr(self, "gecis_s", None) or DV.GECIS_S) / DT)); self.vgecis = self.vgecis_n; self.vk = 0.0; self.vtut = 0
        self.v = 0.0; self.soguma = 0
        if tip.startswith("sicrama"):                       # sıçrama başlangıcı (kök yolu ödülü için)
            self.s_bas = self.env.data.xpos[self.torso][:2].copy(); self.s_yaw = self.yaw(); self.s_odul = False; self.s_olcek = 1.0; self.s_ucus = 0.0; self.s_ucus_en = 0.0

    def sicrama_yolu(self):
        """Sıçrama başından beri gövdenin başlangıç bakış yönünde aldığı yol (m)."""
        f = np.array([np.cos(self.s_yaw), np.sin(self.s_yaw)])
        return float((self.env.data.xpos[self.torso][:2] - self.s_bas) @ f)

    def durum_ilerlet(self):
        """TekEnv/TekVurusEnv durum makineleri (yürüme fazı + vuruş fazı)."""
        ayakta = self.ayakta_mi(); self.ayakta_say = self.ayakta_say + 1 if ayakta else 0
        if self.vtip is not None:
            T_ = len(self.vref["Q"])
            if self.vgecis > 0: self.vgecis -= 1
            else: self.vk = min(self.vk + G.FRAME_SKIP * G.PHYS_DT / self.vref["dt"], T_ - 1)
            if (self.vk >= T_ - 1 and self.vgecis == 0):
                self.vtut += 1
                if self.vtut * DT >= TV.VURUS_TUT_S: self.vtip = None
            if self.dustu_mu(): self.vtip = None
        dustu = self.dustu_mu()
        if self.yuruyor and (self.v <= 0 or dustu): self.yuruyor = False; self.faz_u = 0.0
        elif not self.yuruyor and self.v > 0 and self.ayakta_say >= T.AYAKTA_N and self.vtip is None:
            self.yuruyor = True; self.faz_u = 0.0
        if self.yuruyor:
            C = len(self.env._D[self.dongu_i][0]); self.faz_u = (self.faz_u + G.FRAME_SKIP / C) % 1.0


def ag_yukle(ck):
    """Donuk alt ağ → (ağ, obs ortalama, obs std). Ağ eski robotunsa (109 gözlem) bu robota (govde2: 119) genişletilir;
    yeni eklemler nötr hedefte (tek_env.genislet_robot_genel, 2026-10-21)."""
    import os, tempfile, selfplay_training as st
    n_ck = len(torch.load(ck, map_location="cpu", weights_only=False)["obs_rms_mean"])
    if n_ck > TV.OBS_DIM:                                       # rakip-gözlemli / sıçramalı ağ (2026-10-25; ek boyutlar sonda)
        a = st.CUDAPPOAgent(n_ck, T.ACT_DIM, name="alt"); a.load(ck)
    elif n_ck != TV.OBS_DIM:
        a = T.genislet_robot_genel(ck, os.path.join(tempfile.gettempdir(), f"alt_ag_{os.getpid()}.pth"), n_ck, mutlak=True)
    else:
        a = st.CUDAPPOAgent(TV.OBS_DIM, T.ACT_DIM, name="alt"); a.load(ck)
    net = a.network.cpu().eval()
    return (net, torch.as_tensor(a.obs_rms.mean, dtype=torch.float32),
            torch.as_tensor(np.sqrt(a.obs_rms.var) + 1e-8, dtype=torch.float32))


class DovusMacEnv:
    """reset() → (üst gözlem p1, üst gözlem p2); adim(k1, k2) → (o1, o2, r1, r2, bitti, bilgi)."""
    def __init__(self, alt_ck="TEK_MODELI_v2.pth", env_id=0, xml=None):
        import selfplay_training as st
        self.base = TV.TekVurusEnv(env_id=env_id, xml=xml)    # xml: robot + mekân parçası (taktik_env, 2026-10-22)              # model/data, yürüme döngüleri (_D), _get_obs(p)
        self.model, self.data = self.base.model, self.base.data
        self._D = self.base._D
        self.P = [Oyuncu(self, "p1"), Oyuncu(self, "p2")]
        self.P[0].rakip, self.P[1].rakip = self.P[1], self.P[0]
        self._net, self._mu, self._sd = ag_yukle(alt_ck)

    def _get_obs(self, p):
        self.base.timestep = 0; return self.base._get_obs(p)

    def reset(self, seed=None):
        if seed is not None: np.random.seed(seed)
        mujoco.mj_resetData(self.model, self.data)
        # YÖN (2026-10-21, ölçüldü): rastgele yönde başlayınca robot rakibe dönemeden yürüyüp 18–22 m uzaklaştı (TEK_MODELI_v2
        # dönmesi ~2–3°/s). Başlangıç yön sapması BAS_YON_MAKS ile sınırlı (müfredat: dönme öğrenildikçe büyütülür).
        mesafe = np.random.uniform(*BAS_MESAFE); a = np.random.uniform(-np.pi, np.pi)
        x1, y1 = -0.5 * mesafe * np.cos(a), -0.5 * mesafe * np.sin(a)
        self.P[0].sifirla(x1, y1, a + np.random.uniform(-BAS_YON_MAKS, BAS_YON_MAKS))
        self.P[1].sifirla(-x1, -y1, a + np.pi + np.random.uniform(-BAS_YON_MAKS, BAS_YON_MAKS))
        mujoco.mj_forward(self.model, self.data)
        self.t = 0; self.istat = dict(isabet=[0, 0], hasar=[0.0, 0.0], devrilme=[0, 0], ayakta=[0, 0])
        self._alt_o = [p.alt_gozlem() for p in self.P]
        return self.ust_gozlem(0), self.ust_gozlem(1)

    def _komut_uygula(self, i, k):
        p, r = self.P[i], self.P[1 - i]
        p.son_komut = k
        if p.vtip is not None: return
        hedef = np.arctan2(r.env.data.xpos[r.torso][1] - p.env.data.xpos[p.torso][1],
                           r.env.data.xpos[r.torso][0] - p.env.data.xpos[p.torso][0])
        ad = KOMUTLAR[k]
        if ad in ("ilerle", "yavas", "dur"):
            p.v = {"ilerle": 0.7, "yavas": 0.35, "dur": 0.0}[ad]; p.psi = hedef
        elif ad in ("sola", "saga"):
            p.v = 0.0; p.psi = p.yaw() + (np.pi / 4 if ad == "sola" else -np.pi / 4)
        elif p.ayakta_mi():
            p.vurus_baslat(ad)

    def _temaslar(self):
        """Bu alt adımdaki isabetler: [(saldıran i, hasar, tip, hedef ağırlığı)] (ağırlık ≥ 1 = temiz: baş/göğüs)."""
        d, m = self.data, self.model; out = []; f6 = np.zeros(6)
        for c in range(d.ncon):
            con = d.contact[c]; b1, b2 = m.geom_bodyid[con.geom1], m.geom_bodyid[con.geom2]
            for i in (0, 1):
                a, s = self.P[i], self.P[1 - i]
                for x, y in ((b1, b2), (b2, b1)):
                    # yalnız VURUŞ HAREKETİNDEYKEN (2026-10-21, ölçüldü: pasif rakibin duran yumruğuna yürüyerek çarpmak
                    # 'alınan isabet' sayılıyordu)
                    if x in a.vuran and y in s.hedef_govde and a.soguma <= 0 and a.vtip is not None:
                        mujoco.mj_contactForce(m, d, c, f6); F = float(np.linalg.norm(f6[:3]))
                        if F > HASAR_F_MIN:
                            out.append((i, min(F / 80.0, 8.0) * s.hedef_govde[y], a.vuran_tip[x], s.hedef_govde[y]))
                            a.soguma = int(HASAR_SOGUMA_S / DT)
        return out

    def adim(self, k1, k2):
        self._komut_uygula(0, k1); self._komut_uygula(1, k2)
        r = [0.0, 0.0]; bilgi = {}
        for _ in range(KARAR_ADIM):
            X = torch.as_tensor(np.stack(self._alt_o))
            with torch.no_grad():
                a = self._net.eylem((kes(X, self._mu) - self._mu) / self._sd)[0].numpy()     # deterministik, iki robot tek grup
            hed = [self.P[i].mutlak_hedef(a[i]) for i in (0, 1)]
            d = self.data
            for _s in range(G.FRAME_SKIP):
                for i, p in enumerate(self.P):
                    d.ctrl[p.akt] = np.clip((p.kp * (hed[i] - d.qpos[p.qa]) - p.kd * d.qvel[p.va]) / p.g, -1, 1)
                mujoco.mj_step(self.model, d)
                for i, h, tip, _w in self._temaslar():
                    s = self.P[1 - i]; s.hp = max(0.0, s.hp - h)
                    r[i] += W_HASAR * h; r[1 - i] -= W_HASAR * h
                    self.istat["isabet"][i] += 1; self.istat["hasar"][i] += h
            self.t += 1
            for i, p in enumerate(self.P):
                p.soguma -= 1; p.durum_ilerlet()
                ay = p.ayakta_mi(); self.istat["ayakta"][i] += ay
                if p.ayakta_onceki and p.dustu_mu():
                    p.devrilme += 1; self.istat["devrilme"][i] += 1; r[i] -= W_DEVIR; r[1 - i] += W_DEVIR
                p.ayakta_onceki = not p.dustu_mu()
                p.yerde = 0 if ay else p.yerde + 1
            self._alt_o = [p.alt_gozlem() for p in self.P]
        for i in (0, 1):                                      # şekillendirme (küçük)
            p, q = self.P[i], self.P[1 - i]
            dx = q.env.data.xpos[q.torso][:2] - p.env.data.xpos[p.torso][:2]; dist = float(np.linalg.norm(dx))
            don = np.cos(_sar(np.arctan2(dx[1], dx[0]) - p.yaw()))
            r[i] += W_DONUK * don + W_MESAFE * float(np.exp(-4 * (dist - 0.6) ** 2)) - W_PASIF
        bitti = (self.t * DT >= MAC_SURE_S or any(p.devrilme >= DEVRILME_MAKS for p in self.P)
                 or any(p.yerde * DT >= YERDE_MAKS_S for p in self.P))
        return self.ust_gozlem(0), self.ust_gozlem(1), r[0], r[1], bitti, self.istat

    def ust_gozlem(self, i):
        p, q = self.P[i], self.P[1 - i]; d = self.data
        y = p.yaw(); c, s = np.cos(y), np.sin(y)
        dx = d.xpos[q.torso][:2] - d.xpos[p.torso][:2]; rel = np.array([c * dx[0] + s * dx[1], -s * dx[0] + c * dx[1]])
        dist = float(np.linalg.norm(dx)); acı = np.arctan2(rel[1], rel[0]); dyaw = _sar(q.yaw() - y)
        dv = d.qvel[q.vs:q.vs + 2] - d.qvel[p.vs:p.vs + 2]; rv = np.array([c * dv[0] + s * dv[1], -s * dv[0] + c * dv[1]])
        vur = lambda o: ([1.0, 0.0] if o.vtip == "yumruk" else [0.0, 1.0] if o.vtip == "tekme" else [0.0, 0.0]) + \
                        [(o.vk / len(o.vref["Q"])) if o.vtip else 0.0]
        sk = np.zeros(len(KOMUTLAR)); sk[p.son_komut] = 1
        return np.r_[rel / 3.0, dist / 3.0, np.sin(acı), np.cos(acı), np.sin(dyaw), np.cos(dyaw), rv,
                     float(p.ayakta_mi()), float(q.ayakta_mi()), vur(p), vur(q), p.hp / 100.0, q.hp / 100.0, sk,
                     self.t * DT / MAC_SURE_S].astype(np.float32)
