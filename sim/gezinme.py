# -*- coding: utf-8 -*-
"""Gezinme (PLAN: kesintisiz bölgelerde dövüş — Faz 2, 2026-10-22): robot bölge zincirinde (orman → sur gediği → avlu →
tapınak) yol bulup yürür. Hiyerarşide ORTA katman: taktik ağın 'kovala / kaç / geçide git' komutlarını yürütecek kod.

  * sahne_xml(robot_xml, parca)  : robot dosyasına mekân parçasını (arena/insa/…_komsuN_fizik.xml) ekleyen geçici dosya
  * Harita(model)                 : mekan_* geomlarından dünya doluluk ızgarası (0.25 m), robot yarıçapı kadar şişirilmiş;
                                    yalnız robot boyunu kesen geomlar engel (tavan, lento sayılmaz). A* + görüş hattı sadeleştirme.
  * YolTakip                      : ara noktaları izler; Faz 0'da ölçüldü — yerinde 180° dönüş 6/6 DÜŞÜRÜYORDU, bu yüzden
                                    'yay' kuralı: hedef yön en fazla 45° önde, büyük hatada bile yavaş yürümeye devam.
Çalıştır (ölçüm):  set ROBOT=govde2 & venv\\Scripts\\python.exe gezinme.py <parca.xml> [model.pth] [N]
"""
import os, sys, math, heapq, tempfile
import numpy as np, mujoco

HUCRE, ROBOT_R = 0.25, 0.45
YAKIN_R, YAKIN_CEZA = 0.9, 4.0     # 2026-10-22 (ölçüldü): kapıdan geçince erken dönüp duvar yüzüne sürtünüp takılıyordu
                                   # (avlu→tapınak 1/4) — engele 0.9 m'den yakın hücre ek maliyetli: yol kapı ortasından
YAY_MAKS = math.radians(45.0)
VARIS_M = 0.8


def sahne_xml(robot_xml: str, parca_yolu: str) -> str:
    """Robot XML + mekân parçası → robot dosyasının klasöründe geçici birleşik dosya (çağıran siler)."""
    s = open(robot_xml, encoding="utf-8").read()
    p = open(parca_yolu, encoding="utf-8").read()
    govde = p.split("<mujocoinclude>")[1].split("</mujocoinclude>")[0]
    i = s.rindex("</worldbody>")
    f = os.path.join(os.path.dirname(os.path.abspath(robot_xml)), f"_sahne_{os.getpid()}.xml")
    open(f, "w", encoding="utf-8").write(s[:i] + govde + s[i:])
    return f


class Harita:
    def __init__(self, m: mujoco.MjModel, z_aralik=(0.05, 1.3)):
        d = mujoco.MjData(m); mujoco.mj_forward(m, d)
        self.engel = []                                            # (cx, cy, hx, hy, açı) dönük kutu
        xs, ys = [], []
        for g in range(m.ngeom):
            ad = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if not ad.startswith("mekan_"):
                continue
            t, sz, p = m.geom_type[g], m.geom_size[g], d.geom_xpos[g]
            R = d.geom_xmat[g].reshape(3, 3)
            if t == mujoco.mjtGeom.mjGEOM_BOX:
                hz = abs(R[2, 0]) * sz[0] + abs(R[2, 1]) * sz[1] + abs(R[2, 2]) * sz[2]
                hx, hy, aci = sz[0], sz[1], math.atan2(R[1, 0], R[0, 0])
            elif t in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
                hz = sz[1] + (sz[0] if t == mujoco.mjtGeom.mjGEOM_CAPSULE else 0); hx = hy = sz[0]; aci = 0.0
            elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
                hz = hx = hy = sz[0]; aci = 0.0
            else:
                continue
            xs += [p[0] - hx - hy, p[0] + hx + hy]; ys += [p[1] - hx - hy, p[1] + hx + hy]
            if p[2] + hz < z_aralik[0] or p[2] - hz > z_aralik[1]:
                continue                                           # tavan / lento / zemin altı
            self.engel.append((p[0], p[1], hx, hy, aci))
        self.x0, self.y0 = min(xs) - 1, min(ys) - 1
        self.nx, self.ny = int((max(xs) + 1 - self.x0) / HUCRE) + 1, int((max(ys) + 1 - self.y0) / HUCRE) + 1
        self.dolu = np.zeros((self.nx, self.ny), bool)
        self.yakin = np.zeros((self.nx, self.ny), bool)
        gx = self.x0 + np.arange(self.nx)[:, None] * HUCRE
        gy = self.y0 + np.arange(self.ny)[None, :] * HUCRE
        for cx, cy, hx, hy, a in self.engel:
            c, s = math.cos(a), math.sin(a)
            lx, ly = (gx - cx) * c + (gy - cy) * s, -(gx - cx) * s + (gy - cy) * c
            dx, dy = np.maximum(np.abs(lx) - hx, 0), np.maximum(np.abs(ly) - hy, 0)
            self.dolu |= dx * dx + dy * dy <= ROBOT_R * ROBOT_R
            self.yakin |= dx * dx + dy * dy <= YAKIN_R * YAKIN_R

    def hucre(self, x, y):
        return int(round((x - self.x0) / HUCRE)), int(round((y - self.y0) / HUCRE))

    def merkez(self, i, j):
        return self.x0 + i * HUCRE, self.y0 + j * HUCRE

    def _bos(self, i, j):
        return 0 <= i < self.nx and 0 <= j < self.ny and not self.dolu[i, j]

    def _en_yakin_bos(self, h):
        if self._bos(*h):
            return h
        for r in range(1, 12):
            for di in range(-r, r + 1):
                for dj in (-r, r):
                    for c in ((h[0] + di, h[1] + dj), (h[0] + dj, h[1] + di)):
                        if self._bos(*c):
                            return c
        return h

    def gorur(self, a, b, yakin_da=False):
        n = int(math.hypot(b[0] - a[0], b[1] - a[1]) / (HUCRE / 2)) + 1
        for t in np.linspace(0, 1, n):
            h = self.hucre(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
            if not self._bos(*h) or (yakin_da and self.yakin[h]):
                return False
        return True

    def yol(self, bas, hedef):
        """A* (8 komşu) → sadeleştirilmiş ara noktalar (bas hariç). Yol yoksa None."""
        b, h = self._en_yakin_bos(self.hucre(*bas)), self._en_yakin_bos(self.hucre(*hedef))
        acik, gel, g = [(0.0, b)], {b: None}, {b: 0.0}
        while acik:
            _, u = heapq.heappop(acik)
            if u == h:
                break
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    v = (u[0] + di, u[1] + dj)
                    if (di or dj) and self._bos(*v):
                        ng = g[u] + math.hypot(di, dj) * (1.0 + YAKIN_CEZA * self.yakin[v])
                        if ng < g.get(v, 1e18):
                            g[v] = ng; gel[v] = u
                            heapq.heappush(acik, (ng + math.hypot(h[0] - v[0], h[1] - v[1]), v))
        if h not in gel:
            return None
        hs, u = [], h
        while u is not None:
            hs.append(self.merkez(*u)); u = gel[u]
        hs = hs[::-1]
        sade, i = [hs[0]], 0
        while i < len(hs) - 1:
            j = len(hs) - 1
            while j > i + 1 and not self.gorur(hs[i], hs[j], yakin_da=True):
                j -= 1
            sade.append(hs[j]); i = j
        return sade[1:]


class YolTakip:
    """Ara nokta takipçisi → (ψ*, v*) komutu. 'yay': yön en fazla YAY_MAKS önde (Faz 0: U dönüşü 0/6 → 6/6)."""
    def __init__(self, harita: Harita, hedef):
        self.h, self.hedef, self.W, self.k = harita, hedef, None, 0

    def komut(self, p, yaw):
        if self.W is None or (self.k < len(self.W) and not self.h.gorur(p, self.W[self.k])):
            self.W = self.h.yol(p, self.hedef) or [self.hedef]; self.k = 0      # (yeniden) planla
        while self.k < len(self.W) - 1 and math.hypot(self.W[self.k][0] - p[0], self.W[self.k][1] - p[1]) < VARIS_M:
            self.k += 1
        w = self.W[self.k]
        fark = (math.atan2(w[1] - p[1], w[0] - p[0]) - yaw + math.pi) % (2 * math.pi) - math.pi
        return yaw + float(np.clip(fark, -YAY_MAKS, YAY_MAKS)), (0.7 if abs(fark) < math.radians(30) else 0.35)

    def vardi(self, p):
        return math.hypot(self.hedef[0] - p[0], self.hedef[1] - p[1]) < VARIS_M


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.getcwd())
    import torch, selfplay_training as st
    st.DEVICE = torch.device("cpu"); st.USE_AMP = False
    import getup_train as G, tek_env as T, tek_vurus_env as TV
    torch.set_num_threads(2)
    PARCA = sys.argv[1]
    CK = sys.argv[2] if len(sys.argv) > 2 else "bulut_sonuc/alt_dovus2_govde2_eppoz6M.pth"
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    xml = sahne_xml(G.XML, PARCA)
    try:
        env = TV.TekVurusEnv(env_id=70, xml=xml)
    finally:
        os.remove(xml)
    m = env.model
    harita = Harita(m)
    site = lambda ad: m.site_pos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, ad)].copy()
    hedefler = [(ad, site(ad)[:2]) for ad in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, i) for i in range(m.nsite))
                if ad and ad.startswith("mekan_") and ad.endswith("_acik_alan")]
    hedefler.sort(key=lambda t: t[1][1])                        # zincir kuzeye ilerler
    bas = site("mekan_baslangic_p1")
    ag = st.CUDAPPOAgent(TV.OBS_DIM, T.ACT_DIM, name="g"); ag.load(CK)
    robot_g = {g for g in range(m.ngeom) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or "").startswith("p1")}
    mekan_g = {g for g in range(m.ngeom) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("mekan_")}
    print(f"model {CK} | {len(harita.engel)} engel, ızgara {harita.nx}×{harita.ny} | hedefler: "
          + ", ".join(f"{a.replace('mekan_', '').replace('_acik_alan', '')}" for a, _ in hedefler[1:]), flush=True)
    for t in range(N):
        np.random.seed(40_000 + t); torch.manual_seed(40_000 + t)
        o, _ = env.reset(options=dict(start="ayakta", v=0.0, psi_rel=0.0, komut_sabit=True, itme=False, vurus=False))
        d = env.data; qs = env.p1_qpos_start
        d.qpos[qs:qs + 2] = bas[:2]; d.qpos[qs + 3:qs + 7] = [math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)]   # kuzeye bak
        if env.p2_qpos_start is not None:
            d.qpos[env.p2_qpos_start:env.p2_qpos_start + 2] = [-60.0, -60.0]                                       # P2 uzağa
        mujoco.mj_forward(m, d); env._prev_com = None; o = env.get_obs_both()[0]
        sonuc, sure0, temas, dus, kim, kalkti = [], 0.0, 0, False, {}, 0
        for ad, hd in hedefler[1:]:
            yt = YolTakip(harita, tuple(hd)); n = 0
            while n * T.DT < 90 and not yt.vardi(d.qpos[qs:qs + 2]):
                env._psi, env._v = yt.komut(tuple(d.qpos[qs:qs + 2]), env.yaw())
                o, *_ = env.step(ag.get_action(o, training=False)[0]); n += 1
                for c in d.contact[:d.ncon]:
                    if (c.geom1 in mekan_g and c.geom2 in robot_g) or (c.geom2 in mekan_g and c.geom1 in robot_g):
                        temas += 1; gm = c.geom1 if c.geom1 in mekan_g else c.geom2
                        kim[gm] = kim.get(gm, 0) + 1
                if T.dustu_mu(env):                  # düştü: dur (v*=0), hareket ağı kalksın (≤ 12 s), sonra yola devam
                    dus = True; kalk = 0
                    while kalk * T.DT < 12 and not env.ayakta_mi():
                        env._v = 0.0
                        o, *_ = env.step(ag.get_action(o, training=False)[0]); kalk += 1; n += 1
                    if not env.ayakta_mi():
                        break
                    kalkti += 1
            ok = yt.vardi(d.qpos[qs:qs + 2])
            sonuc.append(f"{ad.replace('mekan_', '').replace('_acik_alan', '')} {'✓' if ok else '✗'} {n * T.DT:4.1f}s")
            if not ok:
                break
        en = sorted(kim.items(), key=lambda x: -x[1])[:3]
        print(f"deneme {t}: " + " → ".join(sonuc) + f" | düştü {dus} (kalkıp devam: {kalkti}) | engele temas {temas} adım"
              + (f" | en çok: {', '.join(f'{mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)}({k})' for g, k in en)}"
                 f" | son konum ({d.qpos[qs]:.1f}, {d.qpos[qs + 1]:.1f})" if en else ""), flush=True)
