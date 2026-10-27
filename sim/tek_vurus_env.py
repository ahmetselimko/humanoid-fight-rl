# -*- coding: utf-8 -*-
"""Birleşik ince ayar ortamı (2026-10-21, onaylı plan): TekEnv (yürü/dur/dön/itme/düş-kalk) + KOMUTLA VURUŞ.
Gözlem 109 = TekEnv 105 + [vur_yumruk, vur_tekme, sin πu, cos πu] — damitma2.VurusZincir.obs ile AYNI anlam:
  vuruş yok → (0, 0, 0, 1); geçiş (0.3 s, dik duruştan referansın ilk karesine) → u = −0.25·kalan/n → 0; sonra u = k/T.
Vuruş: robot ayaktayken (ve vuruşta değilken) zamanı gelince komut gelir; yürüme komutu askıya alınır (v*=0, yön sabit).
  Kesit bitince 0.5 s denge, sonra eski komut düzeni sürer. Vuruş sırasında ödül = eklem taklidi + uç nokta (vuruş anı
  ±0.25 s'de ağırlık 2.0 → GECİKME doğrudan cezalanır; TEK_MODELI_v2 ~0.3–0.4 s geç vuruyordu) + ayakta − eylem.
Müfredat (zorluk, eğiticiden): < 0.2 vuruş yok · ≥ 0.2 bölümde ~1 · ≥ 0.5 ~2–3 (aralık kısalır).
"""
import numpy as np, mujoco
import getup_train as G
import tek_env as T
import dovus_env as DV

OBS_DIM = T.OBS_DIM + 4
VURUS_ARALIK_S = (2.0, 6.0)
VURUS_TUT_S = 0.5
GECIS_U = 0.25                       # damitma2.GECIS_U ile aynı olmalı (K17)
VURUS_TIPLERI = ("yumruk", "tekme")


class TekVurusEnv(T.TekEnv):
    def __init__(self, env_id=0, xml=None):
        self._vtip = None; self._vgecis = 0; self._vgecis_n = 1; self._vk = 0.0
        super().__init__(env_id=env_id, xml=xml)
        self._VR = {t: DV.dovus_refleri(t) for t in VURUS_TIPLERI}
        m = self.model; B = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        mujoco.mj_resetData(m, self.data); mujoco.mj_forward(m, self.data)
        sb = "right" if self.data.xpos[B("p1_thigh_right")][1] > self.data.xpos[B("p1_thigh_left")][1] else "left"
        sk = "left" if self.data.xpos[B("p1_left_upper_arm")][1] > self.data.xpos[B("p1_right_upper_arm")][1] else "right"
        ters = lambda s: "left" if s == "right" else "right"
        self._uc_govde = {("yumruk", "sol"): B(f"p1_fist_{sk}"), ("yumruk", "sag"): B(f"p1_fist_{ters(sk)}"),
                          ("tekme", "sol"): B(f"p1_foot_{sb}"), ("tekme", "sag"): B(f"p1_foot_{ters(sb)}")}
        self._vgecis_n = int(round(DV.GECIS_S / T.DT))

    # ── gözlem ──────────────────────────────────────────────────────────
    def _vurus_gozlem(self):
        if getattr(self, "_vtip", None) is None:
            return [0.0, 0.0, 0.0, 1.0]
        cmd = [1.0, 0.0] if self._vtip == "yumruk" else [0.0, 1.0]
        if self._vgecis > 0:
            u = -GECIS_U * self._vgecis / self._vgecis_n
        else:
            u = self._vk / len(self._vref["Q"])
        return cmd + [np.sin(np.pi * u), np.cos(np.pi * u)]

    def get_obs_both(self):
        o1, o2 = super().get_obs_both()
        return np.concatenate([o1, self._vurus_gozlem()]).astype(np.float32), o2

    # ── vuruş ───────────────────────────────────────────────────────────
    def _uc_goreli(self, q19=None):
        d = self.data
        if q19 is not None:
            yq, yv = d.qpos.copy(), d.qvel.copy(); self.q19_yaz(q19); mujoco.mj_kinematics(self.model, d)
        b = self._uc_govde[(self._vtip, self._vref["taraf"])]
        R = d.xmat[self.p1_torso_id].reshape(3, 3); p = R.T @ (d.xpos[b] - d.xpos[self.p1_torso_id])
        if q19 is not None:
            d.qpos[:] = yq; d.qvel[:] = yv; mujoco.mj_kinematics(self.model, d)
        return p

    def vurus_baslat(self, tip=None, ref=None):
        self._vtip = tip or VURUS_TIPLERI[np.random.randint(2)]
        R = self._VR[self._vtip]; self._vref = R[np.random.randint(len(R)) if ref is None else int(ref)]
        self._vgecis = self._vgecis_n; self._vk = 0.0; self._vtut = 0; self._vbas_adim = self.timestep
        self._v_onceki = self._v; self._v = 0.0; self.vurus_sayisi += 1; self.uc_hata = []

    def _vurus_bitir(self):
        self._vtip = None; self._v = self._v_onceki
        self._vurus_t = self.timestep * T.DT + self._vurus_aralik()

    def _vurus_aralik(self):
        z = float(self.zorluk); a, b = VURUS_ARALIK_S
        return np.random.uniform(a, b) * (2.0 if z < 0.5 else 1.0)

    def reset(self, seed=None, options=None):
        self._vtip = None
        o = super().reset(seed=seed, options=options)
        opt = options or {}
        self.vurus_sayisi = 0; self.uc_hata = []
        self._vurus_acik = opt.get("vurus", float(self.zorluk) >= 0.2)
        self._vurus_t = opt.get("vurus_t", self._vurus_aralik())
        self._vzorla = (opt.get("vurus_tip"), opt.get("vurus_ref"))
        return self.get_obs_both()[0], {}

    def _yeni_komut(self, v=None, psi_rel=None):
        if getattr(self, "_vtip", None) is not None:     # vuruş sırasında komut değişmez
            self._komut_t = self.timestep * T.DT + 1.0; return
        super()._yeni_komut(v, psi_rel)

    def step(self, action_p1, hedef=None):
        vurusta = self._vtip is not None
        o, r, te, tr, info = super().step(action_p1, hedef=hedef)
        t = self.timestep * T.DT
        if not vurusta and self._vurus_acik and t >= self._vurus_t and self.ayakta_mi() and not self._yuruyor_vurus_engeli():
            self.vurus_baslat(*self._vzorla)
            return self.get_obs_both()[0], r, te, tr, info
        if not vurusta:
            return o, r, te, tr, info
        # vuruş fazı
        Tq = len(self._vref["Q"])
        if self._vgecis > 0:
            self._vgecis -= 1
        else:
            self._vk = min(self._vk + G.FRAME_SKIP * G.PHYS_DT / self._vref["dt"], Tq - 1)
        ref = self._vref["Q"][min(int(self._vk), Tq - 1)].copy()
        if self._vgecis > 0:                              # geçişte hedef: dik duruştan ilk kareye doğrusal
            w = 1.0 - self._vgecis / self._vgecis_n; ref[7:] = w * ref[7:]
        j, z, tt = self._pose_feats(self.q19()); self._mimic_ref_yeni = True; jr, zr, trr = self._pose_feats(ref, ref=True)
        r_taklit = (0.6 * float(np.mean(np.exp(-G.MIMIC_EKLEM_K * (j - jr) ** 2))) + 0.25 * float(np.exp(-40 * (z - zr) ** 2))
                    + 0.15 * float(np.exp(-10 * (tt - trr) ** 2)))
        e = self._uc_goreli() - self._uc_goreli(ref); e2 = float(e @ e); self.uc_hata.append(np.sqrt(e2))
        olay_adim = self._vgecis_n + self._vref["olay"] / T.DT
        pencere = abs((self.timestep - self._vbas_adim) - olay_adim) * T.DT < DV.VURUS_PENCERE_S
        a = np.clip(np.asarray(action_p1, np.float64), -1, 1)
        self.r_parca = dict(v_taklit=DV.W_TAKLIT * r_taklit,
                            v_uc=(DV.W_UC_VURUS if pencere else DV.W_UC) * float(np.exp(-DV.K_UC * e2)),
                            ayakta=DV.W_AYAKTA * float(self.ayakta_mi()), eylem=-DV.W_EYLEM * float(np.sum(a * a)))
        r = float(sum(self.r_parca.values()))
        if self._vk >= Tq - 1 and self._vgecis == 0:
            self._vtut += 1
            if self._vtut * T.DT >= VURUS_TUT_S or not self.ayakta_mi():
                self._vurus_bitir()
        elif not self.ayakta_mi() and T.dustu_mu(self):
            self._vurus_bitir()                           # vuruşta düştü → komut biter, kalkış ödülü devralır
        return self.get_obs_both()[0], r, te, tr, info

    def _yuruyor_vurus_engeli(self):
        return False                                      # şimdilik yürürken de başlayabilir (v*=0'a geçer)


def vurus_olcum(agent, env, N=6):
    """POZ: ayakta (v*=0) dururken t=1.5 s'de vuruş komutu; zamanında isabet ve düşme. tip başına N."""
    import torch
    parca = []
    for tip in VURUS_TIPLERI:
        dus, vur = 0, []
        for i in range(N):
            np.random.seed(40_000 + i); torch.manual_seed(40_000 + i)
            o, _ = env.reset(options=dict(start="ayakta", v=0.0, psi_rel=0.0, komut_sabit=True, itme=False, vurus=True,
                                          vurus_t=1.5, vurus_tip=tip, vurus_ref=i % len(env._VR[tip])))
            for _ in range(int(5.0 / T.DT)):
                o, *_ = env.step(agent.get_action(o, training=False)[0])
            dus += T.dustu_mu(env)
            if env.uc_hata:
                k = int(env._vgecis_n + env._vref["olay"] / T.DT); vur.append(env.uc_hata[min(k, len(env.uc_hata) - 1)] * 100)
        parca.append(f"{tip}: dusen {dus}/{N} isabet<10cm {sum(v < 10 for v in vur)}/{len(vur)} (medyan {np.median(vur) if vur else -1:.0f}cm)")
    np.random.seed(None)
    return "  ".join(parca)
