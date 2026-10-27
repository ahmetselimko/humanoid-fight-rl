# -*- coding: utf-8 -*-
"""Canavar görevi (2026-10-24, kullanıcı: "sahneye bir canavar ekleyip onu öldürmelerini bekleyebilir miyiz"):
iki govde2 robotu BİRLİKTE bölge zincirindeki bir canavarı öldürür.

Canavar: iri (~2.3 m), betikli MOCAP gövde (fizik kuvveti onu itemez; o robotları iter). En yakın ayaktaki robota
A* yoluyla yavaş yürür (CANAVAR_HIZ < robot), menzile girince kollarını SAVURUR (SAVURMA_S boyunca ±SAVURMA_DER):
kolu değen robota hasar + itme. Canı CANAVAR_CAN; robotların vuruşları (yumruk/tekme, temas kuvveti) can götürür.
Robotlar: TaktikEnv'deki gibi (taktik ağ komutu → gezinme → donuk hareket ağı; kalkış/gezinme seçicileri);
taktik ağ için "rakip" = canavar (rakip_xy / rakip_bilgi). Robotlar birbirine hasar vermez (dost ateşi yok).
Ödül (ileride eğitim için): +W_HASAR·canavara verilen hasar − W_HASAR·alınan − devrilme + küçük şekillendirme;
canavar ölünce ikisine de +W_ODUL_OLDUR.
"""
import math
import os
import tempfile

import mujoco
import numpy as np
import torch

import dovus_mac_env as M
import getup_train as G
import gezinme as GZ
import taktik_env as TK

_e = lambda ad, v: float(os.environ.get(ad, v))       # denge ayarları (degerlendirme/canavar_olc ile taranır)
CANAVAR_CAN = _e("CANAVAR_CAN", "150")
CANAVAR_HIZ = _e("CANAVAR_HIZ", "0.45")                # m/s (robot yürüyüşü ~0.6–0.7)
CANAVAR_DONUS = math.radians(120)      # rad/s
SALDIRI_M = 1.15                       # bu uzaklıkta savurur
SAVURMA_S, SAVURMA_DER, SAVURMA_ARA_S = 0.5, _e("SAVURMA_DER", "55"), _e("SAVURMA_ARA_S", "2.5")
SAVURMA_HASAR = _e("SAVURMA_HASAR", "6")               # savurma başına robot başına
GOREV_SURE_S = float(os.environ.get("GOREV_SURE_S", "90"))
DEVRILME_YENILGI = 6                   # iki robotun toplam devrilmesi bunu geçerse görev başarısız
W_ODUL_OLDUR = 5.0
TAKIM_PAY = 0.5                        # canavara verilen hasar ödülünün bu kadarı arkadaşa da (iş birliği)
OBS_EK = 4                             # arkadaş: göreli konum (ego, /5), uzaklık /5, ayakta
# canavar geomları: ad → (xml, isabet ağırlığı)
CANAVAR_GEOM = {
    "bas": ('type="sphere" pos="0.05 0 2.05" size="0.24"', 2.0),
    "govde": ('type="capsule" fromto="0 0 0.75 0 0 1.6" size="0.36"', 1.0),
    "kol_sol": ('type="capsule" fromto="0 0.5 1.6 0.8 0.58 0.95" size="0.13"', 0.5),
    "kol_sag": ('type="capsule" fromto="0 -0.5 1.6 0.8 -0.58 0.95" size="0.13"', 0.5),
    "bacak_sol": ('type="capsule" fromto="0 0.2 0.08 0 0.2 0.75" size="0.17"', 0.5),
    "bacak_sag": ('type="capsule" fromto="0 -0.2 0.08 0 -0.2 0.75" size="0.17"', 0.5),
}


def _canavarli_parca(parca_xml):
    """Mekân parçasına canavar gövdesini ekler → geçici dosya (çağıran siler)."""
    s = open(parca_xml, encoding="utf-8").read()
    g = "".join(f'<geom name="canavar_{ad}" {x} rgba="0.28 0.42 0.2 1"/>' for ad, (x, _) in CANAVAR_GEOM.items())
    s = s.replace("</mujocoinclude>", f'<body name="canavar" mocap="true" pos="0 0 0">{g}</body></mujocoinclude>')
    f = tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False, encoding="utf-8"); f.write(s); f.close()
    return f.name


class CanavarEnv(TK.TaktikEnv):
    def __init__(self, parca_xml, alt_ck, dunya_klasoru=None, env_id=0, bas_bolge=None):
        gecici = _canavarli_parca(parca_xml)
        try:
            super().__init__(gecici, alt_ck, dunya_klasoru, env_id)
        finally:
            os.remove(gecici)
        m = self.model
        self.cb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "canavar"); self.mocap = m.body_mocapid[self.cb]
        gid = lambda ad: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"canavar_{ad}")
        self.c_agirlik = {gid(ad): w for ad, (_, w) in CANAVAR_GEOM.items()}
        self.c_kol = {gid("kol_sol"), gid("kol_sag")}
        self.bas_bolge = bas_bolge

    # ───────────── canavar durumu
    def _c_koy(self, xy, yaw):
        self.c_xy, self.c_yaw = np.array(xy, float), float(yaw)
        self.data.mocap_pos[self.mocap] = [xy[0], xy[1], 0.0]
        self.data.mocap_quat[self.mocap] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]

    def rakip_xy(self, i):
        return self.c_xy.copy()

    def rakip_bilgi(self, i):
        sav = self.c_savurma > 0
        vur = [1.0, 0.0, 1.0 - self.c_savurma / SAVURMA_S] if sav else [0.0, 0.0, 0.0]
        return self.c_yaw, 1.0, vur, 100.0 * self.c_can / CANAVAR_CAN, self.c_v.copy()

    def _c_hedef(self):
        """en yakın AYAKTAKİ robot (yoksa en yakın)."""
        d = [float(np.linalg.norm(self._xy(i) - self.c_xy)) for i in (0, 1)]
        ayakta = [i for i in (0, 1) if self.P[i].ayakta_mi()]
        return min(ayakta or [0, 1], key=lambda i: d[i]), d

    def _c_adim(self):
        dt = TK.DT
        h, d = self._c_hedef(); hxy = self._xy(h)
        yon = math.atan2(hxy[1] - self.c_xy[1], hxy[0] - self.c_xy[0]); onceki = self.c_xy.copy()
        if self.c_savurma > 0:                                   # savururken yürümez; gövde yaw'ı süpürür
            self.c_savurma -= dt
            k = 1.0 - max(0.0, self.c_savurma) / SAVURMA_S
            yaw = self.c_yaw_bas + math.radians(SAVURMA_DER) * (1 - 2 * k)
            self._c_koy(self.c_xy, yaw)
            if self.c_savurma <= 0:
                self._c_koy(self.c_xy, self.c_yaw_bas)
        else:
            self.c_ara -= dt
            if d[h] <= SALDIRI_M and self.c_ara <= 0:
                self.c_yaw_bas = yon; self.c_savurma = SAVURMA_S; self.c_ara = SAVURMA_ARA_S
                self.c_vurdu = set(); self.istat["c_savurma"] += 1
            elif d[h] > SALDIRI_M * 0.85:
                if self.c_yol is None or math.hypot(*(np.array(self.c_yol.hedef) - hxy)) > 1.0:
                    self.c_yol = GZ.YolTakip(self.harita, tuple(hxy))
                psi, v = self.c_yol.komut(tuple(self.c_xy), self.c_yaw)
                dy = M._sar(psi - self.c_yaw); yaw = self.c_yaw + float(np.clip(dy, -CANAVAR_DONUS * dt, CANAVAR_DONUS * dt))
                hiz = CANAVAR_HIZ * max(0.0, math.cos(dy))
                self._c_koy(self.c_xy + hiz * dt * np.array([math.cos(yaw), math.sin(yaw)]), yaw)
            else:                                                 # menzilde, bekliyor: yüzünü dön
                dy = M._sar(yon - self.c_yaw)
                self._c_koy(self.c_xy, self.c_yaw + float(np.clip(dy, -CANAVAR_DONUS * dt, CANAVAR_DONUS * dt)))
        self.c_v = (self.c_xy - onceki) / dt

    def _c_temaslar(self):
        """→ [(robot i, canavara hasar)] ve [(robot i, canavardan hasar)] bu alt adımda."""
        d, m = self.data, self.model; ver, al = [], []; f6 = np.zeros(6)
        for c in range(d.ncon):
            con = d.contact[c]
            for cg, og in ((con.geom1, con.geom2), (con.geom2, con.geom1)):
                if cg not in self.c_agirlik:
                    continue
                ob = m.geom_bodyid[og]
                for i, p in enumerate(self.P):
                    if ob not in p.govdeler:
                        continue
                    if ob in p.vuran and p.vtip is not None and p.soguma <= 0:
                        mujoco.mj_contactForce(m, d, c, f6); F = float(np.linalg.norm(f6[:3]))
                        if F > M.HASAR_F_MIN:
                            ver.append((i, min(F / 80.0, 8.0) * self.c_agirlik[cg])); p.soguma = int(M.HASAR_SOGUMA_S / TK.DT)
                    if cg in self.c_kol and self.c_savurma > 0 and i not in self.c_vurdu:
                        al.append((i, SAVURMA_HASAR)); self.c_vurdu.add(i)
        return ver, al

    def ust_gozlem(self, i):
        """taktik gözlemi (47, rakip = canavar) + arkadaş (OBS_EK): ağ 47'lik ise ilk 47 kullanılır."""
        o = super().ust_gozlem(i)
        p = self.P[i]; y = p.yaw(); c, s = math.cos(y), math.sin(y)
        dx = self._xy(1 - i) - self._xy(i)
        ek = np.array([(c * dx[0] + s * dx[1]) / 5.0, (-s * dx[0] + c * dx[1]) / 5.0, np.linalg.norm(dx) / 5.0,
                       float(self.P[1 - i].ayakta_mi())], np.float32)
        return np.concatenate([o, ek])

    # ───────────── gezinme ağı seçici: uzaklık robot ↔ canavar
    def _gez_sec(self, a, X):
        if self.gez is None:
            return a
        uz, yk = self._gez_esik; sec = []
        for i, p in enumerate(self.P):
            dd = float(np.linalg.norm(self._xy(i) - self.c_xy))
            self._gezde[i] = i in self.gez_oyuncular and p.vtip is None and dd > (yk if self._gezde[i] else uz)
            if self._gezde[i]:
                sec.append(i); self.istat["gez"][i] += 1
        if sec:
            net, mu, sd = self.gez
            with torch.no_grad():
                ag = net.eylem((M.kes(X, mu) - mu) / sd)[0].numpy()
            for i in sec:
                a[i] = ag[i]
        return a

    # ───────────── bölüm
    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        mujoco.mj_resetData(self.model, self.data)
        adaylar = [k for k, t in enumerate(self.alan_turu) if t == self.bas_bolge] if self.bas_bolge else []
        b = int(np.random.choice(adaylar)) if adaylar else np.random.randint(len(self.alanlar))
        c = self.alanlar[b][1]
        a = np.random.uniform(-np.pi, np.pi)
        cxy = c + 0.8 * np.array([math.cos(a), math.sin(a)])
        robotlar = []
        for j in (0, 1):                                          # iki robot canavarın iki yanında 2.5–3.5 m
            aa = a + math.pi + (j - 0.5) * np.random.uniform(0.6, 1.4)
            robotlar.append(cxy + np.random.uniform(2.5, 3.5) * np.array([math.cos(aa), math.sin(aa)]))
        for j, xy in enumerate(robotlar):
            self._yerlestir(j, xy, math.atan2(cxy[1] - xy[1], cxy[0] - xy[0]) + np.random.uniform(-0.3, 0.3))
        orta = (robotlar[0] + robotlar[1]) / 2
        self.c_savurma, self.c_ara, self.c_yol, self.c_vurdu = 0.0, 1.0, None, set()
        self.c_can, self.c_v = CANAVAR_CAN, np.zeros(2)
        self._c_koy(cxy, math.atan2(orta[1] - cxy[1], orta[0] - cxy[0]))
        mujoco.mj_forward(self.model, self.data)
        self.t = 0
        self.istat = dict(isabet=[0, 0], hasar=[0.0, 0.0], alinan=[0.0, 0.0], devrilme=[0, 0], ayakta=[0, 0],
                          bolge_degisimi=[0, 0], komut=np.zeros((2, TK.N_KOMUT), int), gez=[0, 0], c_savurma=0,
                          c_isabet=[0, 0], oldu=False, sure_s=0.0)
        self._gezde = [False, False]; self._bolge_son = [self.bolge(0), self.bolge(1)]; self.yt = [None, None]
        self._alt_o = [p.alt_gozlem() for p in self.P]
        return self.ust_gozlem(0), self.ust_gozlem(1)

    def adim(self, k1, k2):
        self._komut_uygula(0, k1); self._komut_uygula(1, k2)
        r = [0.0, 0.0]
        for _ in range(M.KARAR_ADIM):
            self._c_adim()
            for i in (0, 1):
                self._yol_guncelle(i)
            self._alt_o = [p.alt_gozlem() for p in self.P]
            X = torch.as_tensor(np.stack(self._alt_o))
            with torch.no_grad():
                a = self._net.eylem((M.kes(X, self._mu) - self._mu) / self._sd)[0].numpy()
            a = self._gez_sec(a, X)
            a = self._kalkis_sec(a)
            hed = [self.P[i].mutlak_hedef(a[i]) for i in (0, 1)]
            d = self.data
            for _s in range(G.FRAME_SKIP):
                for i, p in enumerate(self.P):
                    d.ctrl[p.akt] = np.clip((p.kp * (hed[i] - d.qpos[p.qa]) - p.kd * d.qvel[p.va]) / p.g, -1, 1)
                mujoco.mj_step(self.model, d)
                ver, al = self._c_temaslar()
                for i, h in ver:
                    self.c_can = max(0.0, self.c_can - h); r[i] += M.W_HASAR * h; r[1 - i] += TAKIM_PAY * M.W_HASAR * h
                    self.istat["isabet"][i] += 1; self.istat["hasar"][i] += h
                for i, h in al:
                    p = self.P[i]; p.hp = max(0.0, p.hp - h); r[i] -= M.W_HASAR * h
                    self.istat["alinan"][i] += h; self.istat["c_isabet"][i] += 1
            self.t += 1
            for i, p in enumerate(self.P):
                p.soguma -= 1; p.durum_ilerlet()
                ay = p.ayakta_mi(); self.istat["ayakta"][i] += ay
                if p.ayakta_onceki and p.dustu_mu():
                    p.devrilme += 1; self.istat["devrilme"][i] += 1; r[i] -= M.W_DEVIR
                p.ayakta_onceki = not p.dustu_mu()
                p.yerde = 0 if ay else p.yerde + 1
            if self.c_can <= 0:
                break
        for i in (0, 1):
            b = self.bolge(i)
            if b != self._bolge_son[i]:
                self.istat["bolge_degisimi"][i] += 1; self._bolge_son[i] = b
            p = self.P[i]; dx = self.c_xy - self._xy(i); dist = float(np.linalg.norm(dx))
            don = math.cos(M._sar(math.atan2(dx[1], dx[0]) - p.yaw()))
            r[i] += M.W_DONUK * don + M.W_MESAFE * float(np.exp(-4 * (dist - 0.9) ** 2)) - M.W_PASIF
        oldu = self.c_can <= 0
        if oldu:
            r = [x + W_ODUL_OLDUR for x in r]
        self.istat["oldu"] = oldu; self.istat["sure_s"] = self.t * TK.DT
        bitti = (oldu or self.t * TK.DT >= GOREV_SURE_S
                 or sum(p.devrilme for p in self.P) >= DEVRILME_YENILGI
                 or all(p.yerde * TK.DT >= M.YERDE_MAKS_S for p in self.P))
        return self.ust_gozlem(0), self.ust_gozlem(1), r[0], r[1], bitti, self.istat
