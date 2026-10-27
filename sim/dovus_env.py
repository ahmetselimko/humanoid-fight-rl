# -*- coding: utf-8 -*-
"""Dövüş hareketi UZMAN ortamı (2026-10-19; onaylı yol: her hareket ayrı uzman → tek ağa damıtma → dövüş).
Yürüme reçetesi (yurume_env.YurumeEnv) tek seferlik harekete uyarlanmıştır:
  * taban = referans kesitin o anki karesi (degerlendirme/dovus_referans.py; referanslar/dovus_secim.json), artık PD eylem
  * faz zamanla ilerler; kesit bitince son kare 1 s tutulur (vuruştan sonra dengede durma)
  * gözlem 102 = GetupEnv 100 + ilerleme (sin πu, cos πu) — yürüme uzmanıyla aynı boyut (damıtmaya hazır)
  * ödül: taklit (eklem + kök yükseklik + eğim, yürüme formülü) + UÇ NOKTA takibi (vuran yumruk/ayak gövdeye göre konumu —
    vuruşun asıl ölçüsü) + ayakta + eylem cezası; düşünce bölüm biter
  * simetri kaybı YOK: referanslar taraflı (sol yumruk / sağ tekme); diğer taraf sonra ayna veriyle
Seçim: ortam değişkeni DOVUS_TIP = yumruk | tekme.
Başlangıç: 'bas' (kesitin ilk karesi, hızsız) %60 · 'rsi' (rastgele kare, referans hızlarıyla) %40.
"""
import os, json
import numpy as np, mujoco
import getup_train as G

DOVUS_TIP = os.environ.get("DOVUS_TIP", "yumruk")
DOVUS_START = dict(bas=0.4, rsi=0.2, ayakta_vurus=0.4)   # 2026-10-21: + ayakta duruştan vuruş (uzmanlar 11/15, 6/9 idi)
TUT_S = 1.0
TABAN_ONDE = 2                       # kare (0.016 s): taban fazın biraz önünde (PD gecikmesi; yürümeyle aynı)
DUSME_Z = 0.55 + G.GOVDE_EK
W_TAKLIT, W_UC, W_AYAKTA, W_EYLEM = 0.6, 0.6, 0.2, 0.01
K_UC = 40.0                          # exp(−K·|Δp|²): 10 cm hata → 0.67, 20 cm → 0.20
# VURUŞ PENCERESİ (2026-10-19, ölçüldü: tekme uzmanı 2M'de 9/9 düşmeden ama ayak hedefe 20–33 cm uzak — bacağı kaldırmıyor
# (ayak ref'ten 10–25 cm alçak, 12–20 cm geride), 'büyük adım' gibi; denge daha kârlı geldi). Uç nokta ödülü vuruş anı
# ±VURUS_PENCERE_S içinde W_UC_VURUS'a çıkar → vuruşu tam yapmak dengeden daha değerli olur.
W_UC_VURUS, VURUS_PENCERE_S = 2.0, 0.25
# HIZ ÖLÇEĞİ (2026-10-19, ölçüldü: referans yumruklar 1.2–1.9 m/s — gerçek düz yumruk ~6–10 m/s; Bandai 'normal' stili
# gösterimlik). DOVUS_HIZ=k → referans zamanı k kat hızlı oynatılır (hızlar ×k). Önce motor gücü fizikte denenmeli.
DOVUS_HIZ = float(os.environ.get("DOVUS_HIZ", "1.0"))

_REF = {}
GECIS_S = float(os.environ.get("DOVUS_GECIS_S", "0.3"))   # ayakta_vurus: ayakta duruştan referansın ilk karesine yumuşak geçiş
# KISA VURUŞ (2026-10-24, kullanıcı: "yumrukları hızlı atamıyorlar"; ölçüldü: vuruş ~1.4 s, yumruk en çok 2.7 m/s):
# VURUS_ONCE_S / VURUS_SONRA_S verilirse referans vuruş anının (olay) bu kadar öncesi/sonrası ile kırpılır (gerçek
# kayıt zamanı, DOVUS_HIZ'dan önce). 0 = kırpma yok.
VURUS_ONCE_S = float(os.environ.get("VURUS_ONCE_S", "0")); VURUS_SONRA_S = float(os.environ.get("VURUS_SONRA_S", "0"))
# GARDLI REFERANS (2026-10-24, kullanıcı: "örnek yumruklarda gard yok muydu"; ölçüldü: Bandai 'normal' yumruklarında atmayan
# el başın ~27 cm altında, gard yok). GARD_REF=1: atmayan kol baştan sona gardda (omuz −70°, dirsek −110°: yumruk çene
# hizasında, yüzün ~15 cm önünde — ileri kinematikle bulundu), atan kol vuruştan önce gardda bekleyip yumruğu oradan çıkarır
# ve vuruştan sonra garda döner; bacaklar baştan sona vuruş anındaki duruşta (bir ayak ~19 cm geride, dizler bükük).
GARD_REF = os.environ.get("GARD_REF", "0") == "1"
GARD_OMUZ, GARD_DIRSEK = np.radians(-70.0), np.radians(-110.0)
_KOL = {"sag": (0, 1), "sol": (2, 3)}                  # ESKI_EKLEMLER: omuz/dirsek sağ, omuz/dirsek sol (Q[:, 7 + i])


def _gardli(Q, olay_s, dt, taraf):
    Q = Q.copy(); n = len(Q); k = min(int(round(olay_s / dt)), n - 1)
    vo, vd = _KOL["sag" if taraf == "sag" else "sol"]; ao, ad_ = _KOL["sol" if taraf == "sag" else "sag"]
    Q[:, 7 + ao], Q[:, 7 + ad_] = GARD_OMUZ, GARD_DIRSEK                    # atmayan kol: hep gard
    k_once = max(0, k - int(round(0.2 / dt))); g0 = max(0, k_once - int(round(0.1 / dt)))
    k_don = min(n - 1, k + int(round(0.15 / dt))); g1 = min(n - 1, k_don + int(round(0.15 / dt)))
    for i in range(n):                                                    # atan kol: gard → yumruk → gard
        if i < g0:
            w = 1.0
        elif i < k_once:
            w = 1.0 - (i - g0) / max(1, k_once - g0)
        elif i <= k_don:
            w = 0.0
        else:
            w = min(1.0, (i - k_don) / max(1, g1 - k_don))
        Q[i, 7 + vo] = w * GARD_OMUZ + (1 - w) * Q[i, 7 + vo]
        Q[i, 7 + vd] = w * GARD_DIRSEK + (1 - w) * Q[i, 7 + vd]
    D = _gard_durus()
    if D is None:
        Q[:, 11:19] = Q[k, 11:19]                                         # bacaklar: vuruş anı duruşu
    else:                                                                 # tasarlanmış gard duruşu: atan elin tersi ayak önde
        s_on, s_ar, k_on, k_ar, a_h, a_k, a_b, p0 = D
        on, ar = ("left", "right") if taraf == "sag" else ("right", "left")
        J = {f"{j}_{t}": 7 + i for i, (j, t) in enumerate([("hip_x", "left"), ("hip_y", "left"), ("knee", "left"), ("ankle", "left"),
                                                          ("hip_x", "right"), ("hip_y", "right"), ("knee", "right"), ("ankle", "right")], 4)}
        for t, h, kk in ((on, -s_on, -k_on), (ar, s_ar, -k_ar)):
            Q[:, J[f"hip_x_{t}"]] = 0.0; Q[:, J[f"hip_y_{t}"]] = h; Q[:, J[f"knee_{t}"]] = kk
            Q[:, J[f"ankle_{t}"]] = np.clip(-(a_h * h + a_k * kk + p0) / a_b, np.radians(-34), np.radians(34))
    return Q


_GD = []


def _gard_durus():
    """degerlendirme/gard_durus_tasarla.py çıktısı: (s_on, s_arka, k_on, k_arka, ayak eğimi katsayıları) — fizikte ölçüldü:
    yalnız PD ile 3 s dik, ayaklar ön-arka 0.22 m, tabanlar yatay."""
    if not _GD:
        f = os.path.join(os.path.dirname(os.path.abspath(__file__)), "referanslar", "gard_durus.npy")
        _GD.append(np.load(f) if os.path.exists(f) else None)
    return _GD[0]


# SIÇRAMA (2026-10-25, kullanıcı: hedef yatay ~0.25 m, ayakta iniş): referans kesitler kırpılmaz/gard eklenmez, kendi hızında
# (SICRAMA_HIZ) oynar; kök ileri yolu SICRAMA_M'ye ölçeklenir (işaret korunur: geri sıçrama negatif).
SICRAMA_M = float(os.environ.get("SICRAMA_M", "0.25")); SICRAMA_HIZ = float(os.environ.get("SICRAMA_HIZ", "1.0"))


def dovus_refleri(tip=None, ayar=None):
    """tip başına önbellek (2026-10-21: tek ağ damıtmasında yumruk ve tekme aynı süreçte).
    ayar = (VURUS_ONCE_S, VURUS_SONRA_S, GARD_REF): 2026-10-25 kafa kafaya maçta iki robot farklı referans kullanabilsin
    (verilmezse ortam değişkenlerindeki genel ayar)."""
    tip = tip or DOVUS_TIP
    once, sonra, gard = ayar if ayar is not None else (VURUS_ONCE_S, VURUS_SONRA_S, GARD_REF)
    anahtar = tip if ayar is None else (tip, once, sonra, gard)
    if anahtar not in _REF:
        # 2026-10-25: DOVUS_SECIM ile başka seçim dosyası (fizikte üretilmiş örnekler: referanslar/dovus_secim_fizik.json)
        sec = json.load(open(os.environ.get("DOVUS_SECIM", "referanslar/dovus_secim.json"), encoding="utf-8"))
        for f in sec[tip]:
            f = f.replace("\\", "/")                      # secim.json Windows'ta yazıldı; Linux (bulut) için
            R = np.load(f); Q = R["qpos"].astype(np.float64); dt = float(R["dt"]); olay_s = float(R["olay_s"])
            ek = {k[3:]: R[k].astype(np.float64) for k in R.files if k.startswith("ek_")}
            sicrama = tip.startswith("sicrama")
            # FİZİKTE ÜRETİLMİŞ örnek (*_fizik.npz): robotun gerçek hızında simülasyon kaydı → hızlandırma/kırpma YOK (Ders 19)
            fizik = f.endswith("_fizik.npz")
            if sicrama and not fizik:
                dx = float(Q[-1, 0] - Q[0, 0])
                Q = Q.copy(); Q[:, 0] = Q[0, 0] + (Q[:, 0] - Q[0, 0]) * (SICRAMA_M / max(abs(dx), 1e-3))
            if (once > 0 or sonra > 0) and not sicrama and not fizik:
                i0 = max(0, int(round((olay_s - once) / dt))) if once > 0 else 0
                i1 = min(len(Q), int(round((olay_s + sonra) / dt)) + 1) if sonra > 0 else len(Q)
                Q = Q[i0:i1]; olay_s -= i0 * dt; ek = {k: v[i0:i1] for k, v in ek.items()}
            if gard and tip == "yumruk":
                Q = _gardli(Q, olay_s, dt, str(R["taraf"]))
            hz = 1.0 if fizik else (SICRAMA_HIZ if sicrama else DOVUS_HIZ)
            V = np.zeros((len(Q), 18)); V[1:-1, 0:3] = (Q[2:, 0:3] - Q[:-2, 0:3]) / (2 * dt)
            V[1:-1, 6:] = (Q[2:, 7:] - Q[:-2, 7:]) / (2 * dt)
            _REF.setdefault(anahtar, []).append(dict(Q=Q, V=V * hz, dt=dt / hz, taraf=str(R["taraf"]),
                                                 olay=olay_s / hz, ad=f,
                                                 # 2026-10-22: bel/omuz-x açıları (dovus_referans_ek.py; eski robotta kullanılmaz)
                                                 EK=ek))
    return _REF[anahtar]


class DovusEnv(G.GetupEnv):
    def __init__(self, env_id=0, tip=None):
        self._u = 0.0; self.tip = tip or DOVUS_TIP; self._gecis = 0
        super().__init__(env_id=env_id)
        self._R = dovus_refleri(self.tip)
        m = self.model; B = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        mujoco.mj_resetData(m, self.data); mujoco.mj_forward(m, self.data)
        # FİZİKSEL taraf (bvh_referans ile aynı kural): robot +y = sol
        sol_bacak = "right" if self.data.xpos[B("p1_thigh_right")][1] > self.data.xpos[B("p1_thigh_left")][1] else "left"
        sol_kol = "left" if self.data.xpos[B("p1_left_upper_arm")][1] > self.data.xpos[B("p1_right_upper_arm")][1] else "right"
        ters = lambda s: "left" if s == "right" else "right"
        self._uc_govde = {("yumruk", "sol"): B(f"p1_fist_{sol_kol}"), ("yumruk", "sag"): B(f"p1_fist_{ters(sol_kol)}"),
                          ("tekme", "sol"): B(f"p1_foot_{sol_bacak}"), ("tekme", "sag"): B(f"p1_foot_{ters(sol_bacak)}")}
        self.r_parca = {}

    def get_obs_both(self):
        o1, o2 = super().get_obs_both()
        u = getattr(self, "_u", 0.0)
        return np.concatenate([o1, [np.sin(np.pi * u), np.cos(np.pi * u)]]).astype(np.float32), o2

    def _kare(self, ek=0):
        r = self._ref; return min(int(self._k) + ek, len(r["Q"]) - 1)

    def taban_hedef(self):
        self._pd_hazirla()
        h = self.eski_hedef(self._ref["Q"][self._kare(TABAN_ONDE)]).astype(np.float64)
        if self._gecis > 0:                        # ayakta_vurus: ayakta duruş (eklemler 0) → referans ilk kare
            w = 1.0 - self._gecis / self._gecis_n; h = w * h
        return h

    def _uc_goreli(self, q19=None):
        """Vuran uzvun (yumruk/ayak) gövde çerçevesindeki konumu. q19 verilirse referans poz için kinematik hesap."""
        d = self.data
        if q19 is not None:
            yedek_q, yedek_v = d.qpos.copy(), d.qvel.copy()
            self.q19_yaz(q19); mujoco.mj_kinematics(self.model, d)
        b = self._uc_govde[(self.tip, self._ref["taraf"])]
        R = d.xmat[self.p1_torso_id].reshape(3, 3); p = R.T @ (d.xpos[b] - d.xpos[self.p1_torso_id])
        if q19 is not None:
            d.qpos[:] = yedek_q; d.qvel[:] = yedek_v; mujoco.mj_kinematics(self.model, d)
        return p

    def reset(self, seed=None, options=None):
        G.GetupEnv.reset(self, seed=seed, options={"start": "itme"})       # defter alanları
        mode = (options or {}).get("start") or np.random.choice(list(DOVUS_START), p=list(DOVUS_START.values()))
        self.start_mode = mode
        i = (options or {}).get("ref"); self._ref = self._R[np.random.randint(len(self._R)) if i is None else int(i)]
        Q, V = self._ref["Q"], self._ref["V"]; T = len(Q)
        self._k = 0.0 if mode in ("bas", "ayakta_vurus") else float(np.random.randint(0, max(1, int(T * 0.8))))
        k = int(self._k); d, p1s = self.data, self.p1_qpos_start
        self._gecis_n = int(round(GECIS_S / (G.PHYS_DT * G.FRAME_SKIP))); self._gecis = 0
        if mode == "ayakta_vurus":                 # 2026-10-21: dik duruştan vuruşa (dövüşte 'dururken aniden vur')
            q0 = Q[0].copy(); q0[3:7] = [1, 0, 0, 0]; q0[7:] = 0.0; self.q19_yaz(q0); self._gecis = self._gecis_n
        else:
            self.q19_yaz(Q[k])
        d.qpos[p1s:p1s + 2] = 0.0
        self.v18_yaz(V[k] if mode == "rsi" else np.zeros(18))
        d.ctrl[:] = 0; mujoco.mj_forward(self.model, d)
        for _ in range(4):
            pen = self._p1_min_contact_dist()
            if pen > -0.005: break
            d.qpos[p1s + 2] += -pen - 0.002; mujoco.mj_forward(self.model, d)
        self._x0 = Q[k, 0]
        self.timestep = 0; self._prev_com = None; self._taban_e = None
        self.max_timesteps = int(round(((T - k) * self._ref["dt"] + TUT_S) / (G.PHYS_DT * G.FRAME_SKIP))) + self._gecis
        self._u = self._k / T; self.r_parca = {}; self.uc_hata = []
        return self.get_obs_both()[0], {}

    def step(self, action_p1, hedef=None):
        self.timestep += 1
        a = np.clip(np.asarray(action_p1, np.float64), -1, 1)
        self._pd_hazirla()
        hedef = self.eylem_hedef(a) if hedef is None else np.clip(np.asarray(hedef, np.float64), self._pd_lo, self._pd_hi)
        d, n = self.data, self.p1_actuator_count
        d.ctrl[n:] = 0.0; d.xfrc_applied[self.p1_torso_id] = 0.0
        for _ in range(G.FRAME_SKIP):
            d.ctrl[:n] = np.clip((self._pd_kp * (hedef - d.qpos[self._pd_qa]) - self._pd_kd * d.qvel[self._pd_va]) / self._pd_g, -1, 1)
            mujoco.mj_step(self.model, d)
        T = len(self._ref["Q"])
        if self._gecis > 0:
            self._gecis -= 1                       # geçiş sürerken faz ilerlemez
        else:
            self._k = min(self._k + G.FRAME_SKIP * G.PHYS_DT / self._ref["dt"], T - 1)
        self._u = self._k / T
        ref = self._ref["Q"][self._kare()].copy(); ref[0] -= self._x0
        q = self.q19(); q_rel = q.copy(); q_rel[0] -= 0.0
        j, z, t = self._pose_feats(q); self._mimic_ref_yeni = True; jr, zr, tr = self._pose_feats(ref, ref=True)
        r_taklit = (0.6 * float(np.mean(np.exp(-G.MIMIC_EKLEM_K * (j - jr) ** 2))) + 0.25 * float(np.exp(-40 * (z - zr) ** 2))
                    + 0.15 * float(np.exp(-10 * (t - tr) ** 2)))
        e = self._uc_goreli() - self._uc_goreli(ref); e2 = float(e @ e); self.uc_hata.append(np.sqrt(e2))
        r_uc = float(np.exp(-K_UC * e2))
        vurus = abs(self._k * self._ref["dt"] - self._ref["olay"]) < VURUS_PENCERE_S
        self.r_parca = dict(taklit=W_TAKLIT * r_taklit, uc=(W_UC_VURUS if vurus else W_UC) * r_uc, ayakta=W_AYAKTA,
                            eylem=-W_EYLEM * float(np.sum(a * a)))
        self.mimic_sum += r_taklit
        bas_z = float(self._get_player_position("p1")[2]); dik = float(d.xmat[self.p1_torso_id][8])
        terminated = bas_z < DUSME_Z or dik < 0.5
        truncated = self.timestep >= self.max_timesteps
        return self.get_obs_both()[0], float(sum(self.r_parca.values())), terminated, truncated, {}

    def mesafe(self):
        return 0.0


def dovus_olcum(agent, env, N=None):
    """POZ: her referans bir kez 'bas'tan, deterministik: düşmeden biten / vuruş anında uç nokta hatası (cm)."""
    parca = []
    for mod in ("bas", "ayakta_vurus"):
        tam, vur = 0, []
        for i in range(len(env._R)):
            np.random.seed(30_000 + i); o, _ = env.reset(options={"start": mod, "ref": i}); done = False
            while not done:
                o, _, te, tr, _ = env.step(agent.get_action(o, training=False)[0]); done = te or tr
            tam += bool(tr and not te)
            k = int(env._ref["olay"] / (G.FRAME_SKIP * G.PHYS_DT)) + (env._gecis_n if mod == "ayakta_vurus" else 0)
            if k < len(env.uc_hata): vur.append(env.uc_hata[k] * 100)
        parca.append(f"{env.tip}/{mod}: dusmeden {tam}/{len(env._R)} | isabet medyan {np.median(vur):.0f} cm "
                     f"(<10 cm: {sum(v < 10 for v in vur)}/{len(vur)})")
    np.random.seed(None)
    return "  ".join(parca)
