# -*- coding: utf-8 -*-
"""Yürüme ortamı (yürüme planı Aşama 3, 2026-10-17, kullanıcı onaylı: A yolu — kinematik taban + RL denge).

Aşama 2 (fizikte açık döngü yürüme referansı) başarısız oldu: yürümede her adım tek ayak üstünde düşme anı,
ezbere PD hedefleri + basit kazançlar sağlam yürüme üretemedi. Bu yüzden taban = KİNEMATİK CMU yürüyüş döngüsü
(degerlendirme/bvh_referans.py; yere batma 0, açılar robot sınırlarında); dengeyi politika öğrenir (DeepMimic gibi).

GetupEnv'den miras: robot, PD + ARTIK eylem (hedef = taban + a·ARTIK_PAY), gözlem (100), simetri altyapısı.
Farklar:
  * taban = döngünün o anki karesi; faz ZAMANLA ilerler (faz takibi yok → Ders 13 donması yok)
  * gözlem + [sin φ, cos φ] (102); ayna: φ → φ+π  (sol/sağ yarım döngü kaymalı)
  * düşünce bölüm biter (erken bitiş); 10 s bölüm
  * ödül: taklit + hız + ayakta + yön/yan sapma cezası; kalemler self.r_parca'da (döküm testi için)
Başlangıçlar: yurume_rsi (döngünün rastgele anı, referans hızlarıyla) / ayakta (döngü başı pozu, hızsız) /
kalkis_sonu (kalkış modelinin ayağa kalktığı gerçek durumlar).
"""
import numpy as np, mujoco
import getup_train as G

YURUME_REFS = ("referanslar/yurume_07_01_kin.npz", "referanslar/yurume_07_02_kin.npz",
               "referanslar/yurume_07_06_kin.npz", "referanslar/yurume_08_01_kin.npz")
YURUME_START = dict(yurume_rsi=0.6, ayakta=0.2, kalkis_sonu=0.2)   # yurume4: + kalkis_sonu (yurume3_6M ondan 8/12)
# kalkis_sonu (2026-10-17): kalkış modelinin (KALKIS_MODELI_artik_6M) ayağa kalktıktan 0.5/1.0 s sonraki gerçek durumları
# (degerlendirme/kalkis_sonu_havuz.py; yaw=0, x/y=0). Beceri geçişi: kalkış ağı → yürüme ağı.
KALKIS_SONU_FILE = "referanslar/kalkis_sonu_havuzu.npz"
_MODLAR = ("yurume_rsi", "ayakta", "kalkis_sonu")
YURUME_SURE_S = 10.0
YURUME_TABAN_ONDE = 2          # kare (0.016 s): taban fazın biraz önünde (PD gecikmesi)
DUSME_Z = 0.55 + G.GOVDE_EK    # baş bunun altına inerse / gövde çok eğilirse bölüm biter
W_TAKLIT, W_HIZ, W_AYAKTA = 0.6, 0.3, 0.2
W_YAN, W_YON, W_EYLEM = 0.5, 0.3, 0.01
W_DUZ = 0.2                    # yere basan ayak düz (ref pozları 0.41, parmak ucu politika 0.18; toplamın ~%5'i)

_DONGU = []
_DONGU_SIM = []
_KS = []
SIM_CIFT = ("ankle", "elbow", "hip_y", "knee", "shoulder")   # hip_x referansta 0


def yurume_donguleri(env, simetrik=False):
    """Her kayıttan bir yürüme döngüsü: (Q[C,19] kök x döngü başına sıfırlı, ilerleme/döngü, hız, qvel[C,18]).
    simetrik=True (2026-10-18): sol/sağ eklem çiftleri, L ile yarım döngü kaymış R'nin ortalamasına eşitlenir
    (ölçüldü: 4 kaydın 4'ünde fiziksel sağ diz 4–7° daha bükük → robot sağa kayıyordu). Asıl döngüler değişmez."""
    if simetrik:
        if not _DONGU_SIM:
            m = env.model
            qa = lambda n: 7 + G.ESKI_EKLEMLER.index(n[3:])          # referanslar ESKİ düzende (19/18)
            va = lambda n: 6 + G.ESKI_EKLEMLER.index(n[3:])
            for D, il, h, V in yurume_donguleri(env):
                C = len(D); D2, V2 = D.copy(), V.copy()
                for c in SIM_CIFT:
                    for X, f in ((D2, qa), (V2, va)):
                        l, r = f(f"p1_{c}_left"), f(f"p1_{c}_right")
                        L = 0.5 * (X[:, l] + np.roll(X[:, r], -(C // 2)))
                        X[:, l] = L; X[:, r] = np.roll(L, C // 2)
                _DONGU_SIM.append((D2, il, h, V2))
        return _DONGU_SIM
    if _DONGU:
        return _DONGU
    m = env.model
    B = lambda nm: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, nm)
    mujoco.mj_resetData(m, env.data); mujoco.mj_forward(m, env.data)
    bs = "right" if env.data.xpos[B("p1_thigh_right")][1] > env.data.xpos[B("p1_thigh_left")][1] else "left"
    kal_adr = 7 + G.ESKI_EKLEMLER.index(f"hip_y_{bs}")          # referanslar ESKİ düzende
    for f in YURUME_REFS:
        R = np.load(f); Q = R["qpos"].astype(np.float64); dt = float(R["dt"]); C = int(round(float(R["dongu_s"]) / dt))
        b0 = int(0.1 / dt)
        t0 = int(np.argmin(Q[b0:b0 + C, kal_adr])) + b0          # fiziksel sol kalça en önde (topuk vuruşu civarı)
        if t0 + C + 1 >= len(Q):
            continue
        D = Q[t0:t0 + C].copy(); ilerle = float(Q[t0 + C, 0] - Q[t0, 0]); D[:, 0] -= Q[t0, 0]; D[:, 1] = 0
        V = np.zeros((C, 18))
        for k in range(C):
            a, b = Q[t0 + k - 1], Q[t0 + k + 1]
            V[k, 0] = (b[0] - a[0]) / (2 * dt); V[k, 2] = (b[2] - a[2]) / (2 * dt)
            V[k, 6:] = (b[7:] - a[7:]) / (2 * dt)
        _DONGU.append((D, ilerle, ilerle / (C * dt), V))
    return _DONGU


def yurume_ayna_haritalari():
    """GetupEnv haritası + faz (sin, cos) → (−sin, −cos) [φ → φ+π]."""
    op, os_, ap, as_ = G.ayna_haritalari()
    return np.r_[op, len(op), len(op) + 1], np.r_[os_, -1.0, -1.0].astype(np.float32), ap, as_


class YurumeEnv(G.GetupEnv):
    def __init__(self, env_id=0, xml=None):
        self._faz_u, self._dongu_i = 0.0, 0
        super().__init__(env_id=env_id, **({"xml": xml} if xml else {}))
        self._D = yurume_donguleri(self)
        self.max_timesteps = int(round(YURUME_SURE_S / (G.PHYS_DT * G.FRAME_SKIP)))
        self.r_parca = {}

    # ── gözlem: + faz ─────────────────────────────────────────────────────
    def get_obs_both(self):
        o1, o2 = super().get_obs_both()
        u = getattr(self, "_faz_u", 0.0)
        return np.concatenate([o1, [np.sin(2 * np.pi * u), np.cos(2 * np.pi * u)]]).astype(np.float32), o2

    # ── referans ─────────────────────────────────────────────────────────
    def _ref(self, u, ileri_dongu=0):
        """Faz u (0..1) için referans qpos (kök x: tamamlanan döngüler kadar ilerletilmiş)."""
        D, ilerle, _, _ = self._D[self._dongu_i]; C = len(D)
        k = int(u * C) % C
        q = D[k].copy(); q[0] += ilerle * ileri_dongu
        return q

    def taban_hedef(self):
        self._pd_hazirla()
        D = self._D[self._dongu_i][0]; C = len(D)
        k = (int(self._faz_u * C) + YURUME_TABAN_ONDE) % C
        return self.eski_hedef(D[k]).astype(np.float64)

    # ── reset ────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        G.GetupEnv.reset(self, seed=seed, options={"start": "itme"})     # tüm defter tutma alanları
        mode = (options or {}).get("start") or np.random.choice(list(YURUME_START), p=list(YURUME_START.values()))
        assert mode in _MODLAR, f"bilinmeyen yürüme başlangıcı: {mode}"
        self.start_mode = mode
        self._dongu_i = np.random.randint(len(self._D))
        D, _, hiz, V = self._D[self._dongu_i]; C = len(D)
        self._faz_u = np.random.rand() if mode == "yurume_rsi" else 0.0
        self._dongu_say = 0
        k = int(self._faz_u * C) % C
        d, p1s, vs = self.data, self.p1_qpos_start, self.p1_vel_start
        if mode == "kalkis_sonu":
            if not _KS:
                h = np.load(KALKIS_SONU_FILE); _KS.extend([h["qpos"], h["qvel"]])
            i = np.random.randint(len(_KS[0]))
            self.q19_yaz(_KS[0][i]); self.v18_yaz(_KS[1][i])
        else:
            self.q19_yaz(D[k]); d.qpos[p1s + 1] = 0.0
            self.v18_yaz(V[k] if mode == "yurume_rsi" else np.zeros(18))
        d.ctrl[:] = 0; mujoco.mj_forward(self.model, d)
        for _ in range(4):                                  # zemine gömülmeyi gider
            pen = self._p1_min_contact_dist()
            if pen > -0.005: break
            d.qpos[p1s + 2] += -pen - 0.002; mujoco.mj_forward(self.model, d)
        self.timestep = 0; self._prev_com = None; self._taban_e = None
        self.max_timesteps = int(round(YURUME_SURE_S / (G.PHYS_DT * G.FRAME_SKIP)))
        self.mesafe0 = float(d.qpos[p1s]); self.r_parca = {}
        return self.get_obs_both()[0], {}

    # ── adım ─────────────────────────────────────────────────────────────
    def step(self, action_p1, hedef=None):
        """hedef: verilirse PD hedefi doğrudan uygulanır (damıtma; GetupEnv.step ile aynı sözleşme)."""
        self.timestep += 1
        a = np.clip(np.asarray(action_p1, np.float64), -1, 1)
        self._pd_hazirla()
        hedef = self.eylem_hedef(a) if hedef is None else np.clip(np.asarray(hedef, np.float64), self._pd_lo, self._pd_hi)
        d, n = self.data, self.p1_actuator_count
        d.ctrl[n:] = 0.0
        d.xfrc_applied[self.p1_torso_id] = 0.0
        for _ in range(G.FRAME_SKIP):
            d.ctrl[:n] = np.clip((self._pd_kp * (hedef - d.qpos[self._pd_qa]) - self._pd_kd * d.qvel[self._pd_va]) / self._pd_g, -1, 1)
            mujoco.mj_step(self.model, d)
        D, ilerle, hiz, _ = self._D[self._dongu_i]; C = len(D)
        u_yeni = self._faz_u + G.FRAME_SKIP / C
        if u_yeni >= 1.0: self._dongu_say += 1
        self._faz_u = u_yeni % 1.0
        # ödül
        p1s = self.p1_qpos_start; q = self.q19()
        ref = self._ref(self._faz_u)
        j, z, t = self._pose_feats(q); self._mimic_ref_yeni = True
        jr, zr, tr = self._pose_feats(ref, ref=True)
        # DUZ AYAK TERIMI (2026-10-17): referans duzeltildi (basan ayak −3…0°) ama politika parmak ucunda kaldi (−19°).
        # Olculdu: politikanin BILEK acisi referanstan yalniz +7° farkli — egim bilekten degil GENEL DURUSTAN (one egik
        # govde, bacak acilari) geliyor; bilek taklidi cozmezdi. Fiziksel olcut: YERE BASAN ayak (en alcak noktasi
        # zemine 3 cm'den yakin) dunyaya gore duz olmali. Basan ayak yoksa (havada) terim 1.
        r_duz = self._duz_ayak()
        # (ayri kalem, W_DUZ: taklit icinde %15 payla etkisi ~0.02/adim — aliskanligi kirmaya yetmez)
        r_taklit = (0.6 * float(np.mean(np.exp(-G.MIMIC_EKLEM_K * (j - jr) ** 2))) + 0.25 * float(np.exp(-40 * (z - zr) ** 2))
                    + 0.15 * float(np.exp(-10 * (t - tr) ** 2)))
        vx, vy = float(d.qvel[self.p1_vel_start]), float(d.qvel[self.p1_vel_start + 1])
        r_hiz = float(np.exp(-4.0 * (vx - hiz) ** 2))
        xm = d.xmat[self.p1_torso_id]; yaw = float(np.arctan2(xm[3], xm[0])); dik = float(xm[8])
        bas_z = float(self._get_player_position("p1")[2])
        self.r_parca = dict(taklit=W_TAKLIT * r_taklit, hiz=W_HIZ * r_hiz, ayakta=W_AYAKTA, duz_ayak=W_DUZ * r_duz,
                            yan=-W_YAN * vy * vy, yon=-W_YON * abs(yaw), eylem=-W_EYLEM * float(np.sum(a * a)))
        r = float(sum(self.r_parca.values()))
        self.mimic_sum += r_taklit
        terminated = bas_z < DUSME_Z or dik < 0.5
        truncated = self.timestep >= self.max_timesteps
        return self.get_obs_both()[0], r, terminated, truncated, {}

    def _duz_ayak(self):
        """Yere basan ayakların dünyaya göre eğim ödülü: ort. exp(−10·eğim²); basan ayak yoksa 1."""
        self._pd_hazirla(); d = self.data; m = self.model; r = []
        for g in self._pd_ayak:
            R = d.geom_xmat[g].reshape(3, 3); sz = m.geom_size[g]
            k = np.array([[i, j, l] for i in (-1, 1) for j in (-1, 1) for l in (-1, 1)]) * sz
            if (d.geom_xpos[g] + k @ R.T)[:, 2].min() < 0.03:
                r.append(float(np.exp(-10.0 * np.arctan2(R[2, 0], R[0, 0]) ** 2)))
        return float(np.mean(r)) if r else 1.0

    # ölçüm yardımcıları
    def mesafe(self):
        return float(self.data.qpos[self.p1_qpos_start]) - self.mesafe0

    def yan_sapma(self):
        return float(self.data.qpos[self.p1_qpos_start + 1])


# ── B0 dönme yeteneği denemesi (2026-10-18, kullanıcı onaylı) ─────────────────────────────────────────────
# Robotta kalça YAW eklemi yok; dönme yalnız adım asimetrisi/kaymayla. Taban asimetrisi testi (donme_testi.py)
# belirsizdi: düz yürümeye eğitilmiş öğretmen dönmeye karşı koydu. Burada öğretmen KISA RL ile sabit bir dönme hızına
# ödüllendirilir (gözlem AYNI → YURUME_MODELI_4M'den devam). Simetri kaybı KAPALI (tek yöne dönme aynayla çelişir).
DONME_HEDEF = np.radians(30.0)          # rad/s, sola (+z); karar eşiği 30°/s
# YERİNDE (2026-10-19, kalça yaw planı Adım 3b): DONME_YERINDE=1 → hedef 45°/s, ileri hız hedefi 0 (yerinde adımlayarak dön).
# Eski robot vs kalça yaw'lı robot aynı ödülle karşılaştırılır.
import os as _os
DONME_YERINDE = _os.environ.get("DONME_YERINDE") == "1"
if DONME_YERINDE:
    DONME_HEDEF = np.radians(45.0)
W_DONME = 0.6


class DonmeEnv(YurumeEnv):
    # Anlık açısal hız adım sallantısıyla ±1.5 rad/s salınıyor (ölçüldü) → ödül ~0.3 s'lik üstel ortalamaya (EMA) bağlı.
    def reset(self, seed=None, options=None):
        self._wz_ema = 0.0
        return super().reset(seed=seed, options=options)

    def step(self, action_p1, hedef=None):
        o, r, te, tr, info = super().step(action_p1, hedef=hedef)
        d = self.data; vs = self.p1_vel_start
        wz = float(d.qvel[vs + 5])                                         # serbest eklem açısal hız z (dünya)
        xm = d.xmat[self.p1_torso_id]; yaw = float(np.arctan2(xm[3], xm[0])); c, s = np.cos(yaw), np.sin(yaw)
        vx, vy = float(d.qvel[vs]), float(d.qvel[vs + 1])
        v_ileri, v_yan = c * vx + s * vy, -s * vx + c * vy                  # gövde çerçevesinde
        hiz = 0.0 if DONME_YERINDE else self._D[self._dongu_i][2]
        p = self.r_parca
        self._wz_ema = 0.95 * getattr(self, "_wz_ema", 0.0) + 0.05 * wz
        p["yon"] = W_DONME * float(np.exp(-4.0 * (self._wz_ema - DONME_HEDEF) ** 2))  # düz-git cezası yerine dönme takibi
        p["hiz"] = W_HIZ * float(np.exp(-4.0 * (v_ileri - hiz) ** 2))      # ileri hız GÖVDE yönünde
        p["yan"] = -W_YAN * v_yan * v_yan
        self.wz_son = wz
        return o, float(sum(p.values())), te, tr, info
