# -*- coding: utf-8 -*-
"""TEK AĞ ince ayar ortamı (2026-10-18, kullanıcı onaylı plan B): komutlu hız/yön, durma, itilme, düş-kalk-devam.

Öğrenci (damitma_buyuk_tur13, büyük ağ) MUTLAK PD hedefi üretir → kalkış/yürüme devri için iki ortama gerek yok;
YurumeEnv (→ GetupEnv) fiziği, reset modları, taklit/düz ayak terimleri ve getup ilerleme ölçüsü yeniden kullanılır.

Gözlem (105) = GetupEnv 100 + yürüme fazı (sin, cos) + komut hızı v*/0.7 + yön hatası (sin Δψ, cos Δψ; gövde çerçevesi).
  İlk 103 boyut damıtma öğrencisiyle AYNI anlamda (v=1 'yürü', faz kalkışta 0) → ağ girişi sıfır ağırlıkla genişletilir.
Faz saati: robot 0.5 s ayakta kaldıktan sonra v*>0 ise yürüme başlar (faz 0'dan); düşünce ya da v*=0 olunca durur (faz 0).
Ödül (kesime göre; kalemler self.r_parca):
  ayakta değil  : kalkış ilerlemesi −0.6 + 1.2·prog (getup_reward'ın prog'u; AYAKTA HER ZAMAN DAHA KÂRLI — eski getup ödülü
                  çömelmeye +2.4 veriyordu, yürüme ~1.3 → çömelip kalma tuzağı olurdu)
  yürüyor       : taklit (simetrik referans) + gövde yönünde hız takibi + yön takibi + düz ayak + ayakta − yan hız − eylem
  ayakta, durgun: yerinde durma + diklik + ayakta + (yarım ağırlıkla) yön takibi − eylem
Bölüm 20 s, düşme bölümü bitirmez (kalkıp devam). Komut 2–5 s'de bir değişir; ayaktayken 4–8 s'de bir itme.
"""
import numpy as np, mujoco
import getup_train as G
import yurume_env as Y

_M = G._ayna_model()
_NJ = sum(1 for j in range(_M.njnt) if (mujoco.mj_id2name(_M, mujoco.mjtObj.mjOBJ_JOINT, j) or "").startswith("p1_")
          and _M.jnt_type[j] != 0)
OBS_DIM = 105 + 2 * (_NJ - 12)                 # eski robot 105; kalça yaw'lı (ROBOT=kalcaz) 109
ACT_DIM = sum(1 for i in range(_M.nu) if mujoco.mj_id2name(_M, mujoco.mjtObj.mjOBJ_ACTUATOR, i).startswith("p1_"))
V_OLCEK = 0.7                                  # gözlemde v*/0.7 → damıtma öğrencisinin v=1 'yürü' anlamı korunur
TEK_START = dict(yerde=0.25, oturma=0.10, zincir=0.10, itme=0.10, ayakta=0.20, yurume_rsi=0.25)
TEK_SURE_S = 20.0
KOMUT_ARALIK_S = (2.0, 5.0)
HIZLAR, HIZ_P = (0.0, 0.35, 0.7), (0.25, 0.25, 0.5)
YON_MAKS = np.radians(120.0)
ITME_ARALIK_S, ITME_DV = (4.0, 8.0), (0.2, 0.6)        # m/s anlık hız darbesi (itme başlangıcı 0–0.3)
DT = G.PHYS_DT * G.FRAME_SKIP
AYAKTA_N = int(round(0.5 / DT))
W_TAKLIT, W_HIZ, W_YON, W_DUR, W_DIK = 0.6, 0.3, 0.5, 0.8, 0.3
W_AYAKTA, W_DUZ, W_YAN, W_EYLEM = 0.2, 0.2, 0.5, 0.01
K_YON = 0.8
# DÖNME HIZI ÖDÜLÜ (2026-10-19, ölçüldü: yön-hatası ödülüyle 4M sonra 30/60/90° komutta yalnız +2/+4/+4° dönüyordu; müfredat
# yalnız ayakta kalmaya baktığı için zorluk 1.0'a çıktı ama dönme öğrenilmedi). B0 dönme denemesinde işe yarayan yöntem:
# istenen dönme hızı = clip(K_DONME·Δψ, ±W_DONME_MAKS), ödül = exp(−4·(ω_EMA − ω_istenen)²) (ω rad/s, ~0.3 s EMA).
K_DONME, W_DONME_MAKS = 1.5, np.radians(40.0)
W_DONME_HIZ = 0.6
W_YON = 0.2                                    # yön hatası terimi kalır ama küçük (asıl sinyal dönme hızı)
YON_ESIK_DER = 25.0                            # müfredat: yürürken ortalama |Δψ| bunun altındaysa zorluk artabilir
# MÜFREDAT (2026-10-18, ölçüldü: tam zorlukta öğrenci 20 s'nin yalnız %5–45'inde ayakta — itme/büyük dönüş/yürürken dur
# hepsi birden geliyor, rastgele düşüşten kalkmayı da bilmiyor → zaman yerde boşa geçiyordu). zorluk 0..1 eğiticiden gelir:
#   zorluk < 0.2: DAMITMA KOŞULLARI — komut bölüm boyunca sabit, düz yön, v* ∈ {0 (yalnız kalkış başl., %25), 0.7}
#     (ölçüldü: komutlar 2–5 s'de bir değişince, yavaş/yürürken-dur gelince öğrenci en kolay ayarda bile %42 ayakta);
#   zorluk ≥ 0.2: komut değişimi açık, yön aralığı ±(30°…120°), dur olasılığı 0.05…0.25, yavaş (0.35) yürüme;
#   zorluk ≥ 0.4: itme (dv üst sınırı 0.2…0.6 m/s).
# Uzun düşüş: 5 s kesintisiz ayakta değilse bölüm biter (terminated → gelecek ödülü yok; düşmek kaçış olmaz çünkü ayakta ödül > 0).
DUSUK_MAKS_S = 5.0
# GÖZLEM KİLİDİ (2026-10-18, ölçüldü): rakip (hareketsiz kukla P2) boyutları — can, uzaklık, göreli konum, rakip z, kombo,
# rakip hızı, rakip uzuv/yumruk konumları (35-42, 44-46, 65-82, 89-94) — gözlemde ROBOTUN YÖNÜNE göre döndürülür; damıtmada
# robot neredeyse hiç dönmediği için dar aralıkta kaldılar (ör. var 0.0014) → dönme komutlarında normalizasyon onları 10×+
# büyütüp ağı dağıtıyordu (ilk koşu: KL 8.2, 24/24 düşüş). Rakip yokken anlamsızlar → damıtma ortalamasına kilitlenir
# (referanslar/tek_gozlem_sabit.npz, damitma_veri/tur0'dan). ZAMAN boyutu (43, bölüm süresi oranı) da kilitli: ölçüldü —
# damıtmada yürüme kesimi hep zaman=0'da başlayıp 10 s'de (1.0) bittiği için öğrenci onu 'yürümeye ne zaman başladım'
# saati olarak kullanmış; TekEnv'de kalkıştan sonra yürümeye başlayınca ~2 s'de, 10 s'yi geçince ~11–14 s'de düşüyordu.
_SABIT = np.load("referanslar/tek_gozlem_sabit.npz")
SABIT_IDX_ESKI, SABIT_DEGER = _SABIT["idx"], _SABIT["deger"]
SABIT_IDX = G.eski_gozlem_idx()[SABIT_IDX_ESKI]     # eski düzen indeksleri → bu robotun gözlemi (eski robotta aynı)


def gozlem_kilitle(O):
    """Gözlem(ler)de rakip boyutlarını sabitle (ortam + BC verisi aynı kuralla)."""
    O = np.array(O, np.float32, copy=True); O[..., SABIT_IDX] = SABIT_DEGER
    return O                                    # exp(−K·Δψ²): 45° → 0.61, 90° → 0.14 (geniş çekirdek: büyük hatada da eğim)


def _sar(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class TekEnv(Y.YurumeEnv):
    _v, _psi = 0.0, 0.0                        # base __init__ gözlem boyutunu ölçerken kullanılır
    zorluk = 0.0                               # eğitici worker mesajıyla ayarlar (müfredat)

    def __init__(self, env_id=0, xml=None):
        # xml (2026-10-22, gezinme): robot + mekân parçası birleşik dosya; verilmezse G.XML (davranış aynı)
        super().__init__(env_id=env_id, **({"xml": xml} if xml else {}))
        self._D = Y.yurume_donguleri(self, simetrik=True)
        self.max_timesteps = int(round(TEK_SURE_S / DT))

    # ── gözlem ──────────────────────────────────────────────────────────
    def yaw(self):
        xm = self.data.xmat[self.p1_torso_id]; return float(np.arctan2(xm[3], xm[0]))

    def get_obs_both(self):
        o1, o2 = super().get_obs_both()                         # 102 (100 + faz)
        o1 = gozlem_kilitle(o1)
        dpsi = _sar(self._psi - self.yaw()) if hasattr(self, "data") else 0.0
        return np.concatenate([o1, [self._v / V_OLCEK, np.sin(dpsi), np.cos(dpsi)]]).astype(np.float32), o2

    # ── komut / itme ────────────────────────────────────────────────────
    def _yeni_komut(self, v=None, psi_rel=None):
        z = float(self.zorluk)
        if z < 0.2:                                               # damıtma koşulları
            kalkis = getattr(self, "start_mode", "ayakta") not in ("ayakta", "yurume_rsi")
            vv = 0.0 if (kalkis and np.random.rand() < 0.25) else 0.7
            self._v = vv if v is None else float(v)
            self._psi = self.yaw() + (0.0 if psi_rel is None else float(psi_rel))
            self._komut_t = 1e9                                   # bölüm boyunca sabit
            return
        p_dur = 0.05 + 0.20 * z
        hp = np.array([p_dur, (1 - p_dur) * 0.35, (1 - p_dur) * 0.65])
        self._v = float(np.random.choice(HIZLAR, p=hp)) if v is None else float(v)
        ym = np.radians(30.0) + (YON_MAKS - np.radians(30.0)) * z
        self._psi = self.yaw() + (np.random.uniform(-ym, ym) if psi_rel is None else float(psi_rel))
        self._komut_t = self.timestep * DT + np.random.uniform(*KOMUT_ARALIK_S)

    # ── reset ───────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        o = options or {}
        mode = o.get("start") or np.random.choice(list(TEK_START), p=list(TEK_START.values()))
        if mode in ("ayakta", "yurume_rsi"):
            Y.YurumeEnv.reset(self, options={"start": mode})
        else:
            G.GetupEnv.reset(self, options={"start": mode})
            self._faz_u, self._dongu_i, self._dongu_say = 0.0, np.random.randint(len(self._D)), 0
            self.mesafe0 = float(self.data.qpos[self.p1_qpos_start]); self._taban_e = None
        self.start_mode = mode
        self.timestep = 0; self.max_timesteps = int(round(TEK_SURE_S / DT))
        self._sabit = bool(o.get("komut_sabit", False)); self._itme_acik = bool(o.get("itme", True))
        self._yeni_komut(o.get("v"), o.get("psi_rel"))
        self._itme_t = np.random.uniform(*ITME_ARALIK_S)
        self._ayakta_say = AYAKTA_N if mode in ("ayakta", "yurume_rsi") else 0
        self._yuruyor = mode == "yurume_rsi" and self._v > 0
        if not self._yuruyor: self._faz_u = 0.0
        self.r_parca = {}; self.itme_sayisi = 0; self.ayakta_adim = 0; self._dusuk = 0
        self._wz_ema = 0.0; self.yon_hata_top, self.yon_hata_n = 0.0, 0
        self._prev_com = None
        return self.get_obs_both()[0], {}

    # ── adım ────────────────────────────────────────────────────────────
    def mutlak_hedef(self, a):
        self._pd_hazirla()
        return np.clip(self._pd_orta + np.clip(a, -1, 1) * G.PD_ASIM * self._pd_yari, self._pd_lo, self._pd_hi)

    def ayakta_mi(self):
        return self._get_player_position("p1")[2] > G.STAND_Z and self.data.xmat[self.p1_torso_id][8] > G.STAND_TORSO

    def step(self, action_p1, hedef=None):
        self.timestep += 1
        a = np.clip(np.asarray(action_p1, np.float64), -1, 1)
        hedef = self.mutlak_hedef(a) if hedef is None else np.clip(hedef, self._pd_lo, self._pd_hi)
        d, n, vs = self.data, self.p1_actuator_count, self.p1_vel_start
        d.ctrl[n:] = 0.0; d.xfrc_applied[self.p1_torso_id] = 0.0
        for _ in range(G.FRAME_SKIP):
            d.ctrl[:n] = np.clip((self._pd_kp * (hedef - d.qpos[self._pd_qa]) - self._pd_kd * d.qvel[self._pd_va]) / self._pd_g, -1, 1)
            mujoco.mj_step(self.model, d)
        t = self.timestep * DT
        if not self._sabit and t >= self._komut_t:
            self._yeni_komut()
        ayakta = self.ayakta_mi()
        self._ayakta_say = self._ayakta_say + 1 if ayakta else 0
        self.ayakta_adim += ayakta; self._dusuk = 0 if ayakta else self._dusuk + 1
        if self._itme_acik and self.zorluk >= 0.4 and t >= self._itme_t:
            if ayakta:
                ang = np.random.uniform(-np.pi, np.pi)
                dv = np.random.uniform(ITME_DV[0], ITME_DV[0] + (ITME_DV[1] - ITME_DV[0]) * self.zorluk)
                d.qvel[vs] += dv * np.cos(ang); d.qvel[vs + 1] += dv * np.sin(ang); self.itme_sayisi += 1
            self._itme_t = t + np.random.uniform(*ITME_ARALIK_S)
        # yürüme durumu (histerezis: 0.5 s ayakta + v*>0 → başla; v*=0 ya da düşme → bitir)
        bas_z = float(self._get_player_position("p1")[2]); xm = d.xmat[self.p1_torso_id]; dik = float(xm[8])
        dustu = bas_z < Y.DUSME_Z or dik < 0.5
        if self._yuruyor and (self._v <= 0 or dustu):
            self._yuruyor = False; self._faz_u = 0.0
        elif not self._yuruyor and self._v > 0 and self._ayakta_say >= AYAKTA_N:
            self._yuruyor = True; self._faz_u = 0.0
        if self._yuruyor:
            C = len(self._D[self._dongu_i][0]); self._faz_u = (self._faz_u + G.FRAME_SKIP / C) % 1.0
        # ödül
        yaw = self.yaw(); c, s = np.cos(yaw), np.sin(yaw)
        vx, vy = float(d.qvel[vs]), float(d.qvel[vs + 1]); v_ileri, v_yan = c * vx + s * vy, -s * vx + c * vy
        dpsi = _sar(self._psi - yaw); r_yon = float(np.exp(-K_YON * dpsi * dpsi))
        self._wz_ema = 0.95 * self._wz_ema + 0.05 * float(d.qvel[vs + 5])
        wz_ist = float(np.clip(K_DONME * dpsi, -W_DONME_MAKS, W_DONME_MAKS))
        r_donme = float(np.exp(-4.0 * (self._wz_ema - wz_ist) ** 2))
        if self._yuruyor:
            self.yon_hata_top += abs(dpsi); self.yon_hata_n += 1
        eylem = -W_EYLEM * float(np.sum(a * a))
        if self._yuruyor:
            q = self.q19(); ref = self._ref(self._faz_u)
            j, z, tt = self._pose_feats(q); self._mimic_ref_yeni = True; jr, zr, tr = self._pose_feats(ref, ref=True)
            r_taklit = (0.6 * float(np.mean(np.exp(-G.MIMIC_EKLEM_K * (j - jr) ** 2))) + 0.25 * float(np.exp(-40 * (z - zr) ** 2))
                        + 0.15 * float(np.exp(-10 * (tt - tr) ** 2)))
            self.mimic_sum += r_taklit
            self.r_parca = dict(taklit=W_TAKLIT * (1.0 if self._v >= 0.6 else 0.5) * r_taklit,
                                hiz=W_HIZ * float(np.exp(-4.0 * (v_ileri - self._v) ** 2)), yon=W_YON * r_yon, donme=W_DONME_HIZ * r_donme,
                                ayakta=W_AYAKTA, duz_ayak=W_DUZ * self._duz_ayak(), yan=-W_YAN * v_yan * v_yan, eylem=eylem)
        elif ayakta:
            self.r_parca = dict(dur=W_DUR * float(np.exp(-10.0 * (vx * vx + vy * vy))), dik=W_DIK * dik,
                                yon=0.5 * W_YON * r_yon, donme=0.5 * W_DONME_HIZ * r_donme, ayakta=W_AYAKTA, eylem=eylem)
        else:
            h = min(bas_z / G.STAND_HEAD_Z, 1.0); prog = 0.3 * max(dik, 0.0) + 0.7 * h * h
            flz = float(d.xpos[self.key_body_ids['p1_foot_left']][2]); frz = float(d.xpos[self.key_body_ids['p1_foot_right']][2])
            vz = float(d.qvel[vs + 2]); sakin = abs(vz) < G.CALM_VZ and min(flz, frz) < G.FOOT_PLANTED_Z
            self.r_parca = dict(kalkis=(-0.6 + 1.2 * prog) if sakin else -0.8, eylem=eylem)
        r = float(sum(self.r_parca.values()))
        terminated = self._dusuk * DT >= DUSUK_MAKS_S
        truncated = self.timestep >= self.max_timesteps
        return self.get_obs_both()[0], r, terminated, truncated, {}


def tek_ayna_haritalari():
    """yurume (102: faz φ→φ+π) + v* (değişmez) + yön hatası (sin → −sin, cos aynı; aynada Δψ → −Δψ)."""
    op, os_, ap, as_ = Y.yurume_ayna_haritalari()
    n = len(op)
    return np.r_[op, n, n + 1, n + 2], np.r_[os_, 1.0, -1.0, 1.0].astype(np.float32), ap, as_


def genislet(ck_in, ck_out, ek=OBS_DIM - 103):
    """103 girişli damıtma öğrencisini 105'e genişlet: yeni giriş sütunları SIFIR ağırlık (davranış birebir aynı),
    obs_rms yeni boyutlarda mean 0 / var 1. Aktör ilk katmanı + ikiz critic ilk katmanları."""
    import torch, selfplay_training as st
    eski = st.CUDAPPOAgent(103, ACT_DIM, name="E"); eski.load(ck_in)
    yeni = st.CUDAPPOAgent(103 + ek, ACT_DIM, name="T", gizli=eski.network.gizli)
    sd_e, sd_y = eski.network.state_dict(), yeni.network.state_dict()
    for k, v in sd_e.items():
        if sd_y[k].shape == v.shape:
            sd_y[k] = v.clone()
        elif v.dim() == 2 and sd_y[k].shape[1] == v.shape[1] + ek:      # ilk Linear katmanları
            w = torch.zeros_like(sd_y[k]); w[:, :v.shape[1]] = v; sd_y[k] = w
        else:
            raise ValueError(f"beklenmeyen şekil {k}: {tuple(v.shape)} → {tuple(sd_y[k].shape)}")
    yeni.network.load_state_dict(sd_y)
    yeni.obs_rms.mean = np.r_[eski.obs_rms.mean, np.zeros(ek)]; yeni.obs_rms.var = np.r_[eski.obs_rms.var, np.ones(ek)]
    yeni.obs_rms.count = eski.obs_rms.count
    yeni.save(ck_out)
    return yeni


# ── ölçüm (POZ ve tek_test.py ortak) ────────────────────────────────────────────────────────────────────────
def dustu_mu(env):
    return float(env._get_player_position("p1")[2]) < Y.DUSME_Z or float(env.data.xmat[env.p1_torso_id][8]) < 0.5


def kos(env, ag, sure_s, tohum, itme_t=None, itme_dv=0.6, **opt):
    """Deterministik bölüm. Döner: kayıt sözlüğü (adım başına yaw, konum, ileri hız, yürüyor, düştü)."""
    import torch
    np.random.seed(tohum); torch.manual_seed(tohum)
    o, _ = env.reset(options=dict(opt, komut_sabit=True, itme=False))
    K = dict(yaw=[], xy=[], vi=[], yur=[], dus=[], psi=env._psi)
    for k in range(int(round(sure_s / DT))):
        if itme_t is not None and abs(k * DT - itme_t) < DT / 2:
            ang = np.random.uniform(-np.pi, np.pi); vs = env.p1_vel_start
            env.data.qvel[vs] += itme_dv * np.cos(ang); env.data.qvel[vs + 1] += itme_dv * np.sin(ang)
        o, *_ = env.step(ag.get_action(o, training=False)[0])
        y = env.yaw(); c, s = np.cos(y), np.sin(y); d = env.data; vs = env.p1_vel_start
        K["yaw"].append(y); K["xy"].append(d.qpos[env.p1_qpos_start:env.p1_qpos_start + 2].copy())
        K["vi"].append(c * d.qvel[vs] + s * d.qvel[vs + 1]); K["yur"].append(env._yuruyor); K["dus"].append(dustu_mu(env))
    K["ayakta_son"] = env.ayakta_mi()
    return {k: (np.array(v) if isinstance(v, list) else v) for k, v in K.items()}


def tek_olcum(agent, env, N=6):
    """POZ (eğitim sırasında, kısa): kalkış / düz yürüme / 90° dönüş / dur — tek_test.py'nin küçük örneklemi."""
    kal = sum(kos(env, agent, 10, 20_000 + t, start="yerde", v=0.0, psi_rel=0.0)["ayakta_son"] for t in range(N))
    yur, hz = 0, []
    for t in range(N):
        K = kos(env, agent, 10, 20_100 + t, start="ayakta", v=0.7, psi_rel=0.0); yur += not K["dus"].any(); hz.append(K["vi"][-312:].mean())
    don = []
    for t in range(N):
        K = kos(env, agent, 5, 20_200 + t, start="ayakta", v=0.7, psi_rel=(1 if t % 2 else -1) * np.pi / 2)
        don.append(abs(np.degrees(_sar(K["psi"] - K["yaw"][int(3 / DT)]))))
    # dur: düşen robot da yerinde kalır → 'durdu' = düşmedi VE son 3 s kayma < 0.3 m (2026-10-18 düzeltme)
    dur, dur_ok = [], 0
    for t in range(N):
        K = kos(env, agent, 6, 20_300 + t, start="yurume_rsi", v=0.0, psi_rel=0.0)
        k_ = float(np.linalg.norm(K["xy"][-1] - K["xy"][-int(3 / DT)])); dur.append(k_); dur_ok += (not K["dus"].any()) and k_ < 0.3
    np.random.seed(None)
    return (f"kalkis(yerde)={kal}/{N}  yurume10s={yur}/{N} hiz={np.mean(hz):.2f}  "
            f"donus90_3s_hata={np.median(don):.0f}deg(<15:{sum(x < 15 for x in don)}/{N})  dur(dusmeden,<0.3m)={dur_ok}/{N}")


def genislet_robot(ck_in, ck_out):
    """Eski robot tek ağı (105 gözlem / 12 eylem) → BU robota (ROBOT=kalcaz: 109 / 14). Gözlem sütunları İSİMLE eşlenip
    araya yerleşir (yeni eklem açı/hızları SIFIR ağırlık), yeni eylemler sona eklenir (ağırlık/bias 0 → tanh 0 → mutlak
    hedef = eklem aralığı ortası = 0°). Böylece yeni eklemler 0'da tutulurken davranış eski ağla aynı."""
    import torch, selfplay_training as st
    eski = st.CUDAPPOAgent(105, 12, name="E"); eski.load(ck_in)
    yeni = st.CUDAPPOAgent(OBS_DIM, ACT_DIM, name="R", gizli=eski.network.gizli)
    esle = np.r_[G.eski_gozlem_idx(), OBS_DIM - 5 + np.arange(5)]          # 105 eski → yeni indeksler
    sd_e, sd_y = eski.network.state_dict(), yeni.network.state_dict()
    for k, v in sd_e.items():
        w = sd_y[k]
        if w.shape == v.shape:
            sd_y[k] = v.clone()
        elif v.dim() == 2 and v.shape[1] == 105 and w.shape[1] == OBS_DIM:   # ilk katmanlar (aktör + ikiz critic)
            w = torch.zeros_like(w); w[:, esle] = v; sd_y[k] = w
        elif v.shape[0] == 12 and w.shape[0] == ACT_DIM:                      # aktör son katman / log_std
            w = torch.zeros_like(w) if k != "actor_log_std" else torch.full_like(w, float(v.mean()))
            w[:12] = v; sd_y[k] = w
        else:
            raise ValueError(f"beklenmeyen şekil {k}: {tuple(v.shape)} → {tuple(w.shape)}")
    yeni.network.load_state_dict(sd_y)
    mu, var = np.zeros(OBS_DIM), np.ones(OBS_DIM); mu[esle], var[esle] = eski.obs_rms.mean, eski.obs_rms.var
    yeni.obs_rms.mean, yeni.obs_rms.var, yeni.obs_rms.count = mu, var, eski.obs_rms.count
    yeni.ret_rms = eski.ret_rms                     # ödül ölçeği de taşınır (yoksa critic 112× hatayla başlıyordu)
    yeni.save(ck_out)
    return yeni, esle


def notr_eylem():
    """Mutlak eylem uzayında her p1 motorunu 0° hedefe götüren eylem (aralığı simetrik eklemlerde 0)."""
    m = _M; aj = [m.actuator_trnid[i, 0] for i in range(m.nu) if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i).startswith("p1_")]
    lo, hi = m.jnt_range[aj, 0], m.jnt_range[aj, 1]
    return np.clip(-0.5 * (lo + hi) / (G.PD_ASIM * 0.5 * (hi - lo)), -1, 1)


def bc_robota_uyarla(O, L, n_eski=109):
    """Eski robotun öğretmen verisi (gözlem n_eski = 100 + ek, etiket 12 mutlak eylem) → bu robot (2026-10-21, govde2).
    Gözlem sütunları genişletmedeki gibi İSİMLE yerleşir; yeni eklem açı/hızları 0 (nötr duruş, durağan); yeni etiketler
    notr_eylem() (öğretmen yeni eklemleri nötrde tutar = eşdeğerlik testinde ölçülen davranış)."""
    ek = n_eski - 100; n_yeni = len(G.eski_gozlem_idx()) + 2 * (_NJ - 12) + ek
    esle = np.r_[G.eski_gozlem_idx(), n_yeni - ek + np.arange(ek)]
    O2 = np.zeros((len(O), n_yeni), np.float32); O2[:, esle] = O
    L2 = np.tile(notr_eylem().astype(L.dtype), (len(L), 1)); L2[:, :12] = L
    return O2, L2


def genislet_robot_genel(ck_in, ck_out, n_eski, mutlak=False):
    """Eski robotun HERHANGİ bir ağı (gözlem = 100 GetupEnv + n_eski−100 ek; eylem 12) → bu robota. genislet_robot'un genel hali
    (ör. yürüme öğretmeni 102 → 106). Ek boyutlar (faz vb.) sonda kalır; yeni eylemler 0 (artık modda taban + 0)."""
    import torch, selfplay_training as st
    ek = n_eski - 100
    eski = st.CUDAPPOAgent(n_eski, 12, name="E"); eski.load(ck_in)
    n_yeni = len(G.eski_gozlem_idx()) + 2 * (_NJ - 12) + ek
    yeni = st.CUDAPPOAgent(n_yeni, ACT_DIM, name="R", gizli=eski.network.gizli)
    esle = np.r_[G.eski_gozlem_idx(), n_yeni - ek + np.arange(ek)]
    sd_e, sd_y = eski.network.state_dict(), yeni.network.state_dict()
    for k, v in sd_e.items():
        w = sd_y[k]
        if w.shape == v.shape:
            sd_y[k] = v.clone()
        elif v.dim() == 2 and v.shape[1] == n_eski and w.shape[1] == n_yeni:
            w = torch.zeros_like(w); w[:, esle] = v; sd_y[k] = w
        elif v.shape[0] == 12 and w.shape[0] == ACT_DIM:
            w = torch.zeros_like(w) if k != "actor_log_std" else torch.full_like(w, float(v.mean()))
            w[:12] = v; sd_y[k] = w
        else:
            raise ValueError(f"beklenmeyen şekil {k}: {tuple(v.shape)} → {tuple(w.shape)}")
    # 2026-10-21 (govde2): aralığı simetrik OLMAYAN yeni eklemlerde (omuz yana açma −0.3…1.5) tanh(0) = aralık ortası ≠ 0°.
    # Mutlak eylemde hedef = orta + a·PD_ASIM·yarı → a = −orta/(PD_ASIM·yarı) için aktör son katman bias'ı atanh(a).
    # (Artık modda hedef = taban + a → a = 0 doğru; bias'a dokunulmaz.) Simetrik aralıkta (kalça yaw, bel) bias 0 kalır.
    if mutlak:                                     # TekEnv/TekVurusEnv ağları (TEK_MODELI_*, alt_dovus_*)
        m = _M; aj = [m.actuator_trnid[i, 0] for i in range(m.nu) if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i).startswith("p1_")]
        lo, hi = m.jnt_range[aj, 0], m.jnt_range[aj, 1]; orta, yari = 0.5 * (lo + hi), 0.5 * (hi - lo)
        b = sd_y["actor_last.bias"]
        for j in range(12, ACT_DIM):
            b[j] = float(np.arctanh(np.clip(-orta[j] / (G.PD_ASIM * yari[j]), -0.999, 0.999)))
        sd_y["actor_mean.2.bias"] = b.clone()       # aynı katman state_dict'te iki adla (actor_last = actor_mean[2])
    yeni.network.load_state_dict(sd_y)
    mu, var = np.zeros(n_yeni), np.ones(n_yeni); mu[esle], var[esle] = eski.obs_rms.mean, eski.obs_rms.var
    yeni.obs_rms.mean, yeni.obs_rms.var, yeni.obs_rms.count = mu, var, eski.obs_rms.count
    yeni.ret_rms = eski.ret_rms                     # ödül ölçeği de taşınır (yoksa critic 112× hatayla başlıyordu)
    yeni.save(ck_out)
    return yeni
