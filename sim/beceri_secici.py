# -*- coding: utf-8 -*-
"""Beceri seçici (2026-10-22): ayaktayken DÖVÜŞ ağı, düşünce KALKIŞ uzmanı (hiyerarşinin orta katmanı).
Ölçüldü: dövüş ince ayarları uzadıkça kalkış kayboluyor (v3: sırtüstü 0/12; 2M kaydında 6/8 idi) — BC ve yerde başlama
yetmedi. Kalkışı dövüş ağına zorla tutturmak yerine, robot yerdeyken kontrol kalkışı en iyi bilen modele devredilir.
  BeceriSecici(dovus_ag, kalkis_ag).eylem(gozlem, ayakta) → eylem
Kurallar: ayakta değilse kalkış uzmanı; ayağa kalkınca DONUS_S kadar daha uzmanda kalır (titreşmesin), sonra dövüş ağı.
  set ROBOT=govde2 & set DOVUS_HIZ=1.5 & set KOL_KP=3 & venv\\Scripts\\python.exe beceri_secici.py <dovus.pth> <kalkis.pth>
"""
import os, sys
import numpy as np

DONUS_S = 0.3


class DonmeHareketi:
    """Açık döngü yuvarlanma (degerlendirme/donme_ara.py, CEM ile bulundu: gerçek yüzüstü düşüşlerin 42/43'ünde
    sırtüstüne çeviriyor; RL dönme uzmanı 2M'de 11/43). Ara pozlar sırayla tut_s süre tutulur; dönme_ag yerine verilir."""

    def __init__(self, yol, dt):
        h = np.load(yol); self.pozlar = h["pozlar"]; self.adim = int(round(float(h["tut_s"]) / dt)); self.t = 0

    def sifirla(self):
        self.t = 0

    def get_action(self, o, training=False):
        a = self.pozlar[min(self.t // self.adim, len(self.pozlar) - 1)]; self.t += 1
        return a, None


class BeceriSecici:
    def __init__(self, dovus_ag, kalkis_ag, dt, donme_ag=None):
        # 2026-10-22: donme_ag (yüzüstü → sırtüstü dönme uzmanı) verilirse YÜZÜSTÜ yatarken o sürer
        self.d, self.k, self.y = dovus_ag, kalkis_ag, donme_ag; self.yuzustu = False
        self.n_donus = int(round(DONUS_S / dt)); self.sayac = self.n_donus; self.kalkista = False
        self.gecis = 0

    def sifirla(self):
        self.sayac = self.n_donus; self.kalkista = False

    def eylem(self, o, ayakta, ileri_z=None):
        """ileri_z: gövde ileri ekseninin dünya-z bileşeni (< −0.3 = yüzüstü); verilmezse dönme uzmanı kullanılmaz."""
        self.yuzustu = (self.y is not None and ileri_z is not None and not ayakta and ileri_z < -0.3)
        if self.yuzustu:
            self.kalkista, self.sayac = True, 0
            return self.y.get_action(o, training=False)[0]
        if hasattr(self.y, "sifirla"):                      # yuvarlanma hareketi her yüzüstü yatışta baştan başlar
            self.y.sifirla()
        if not ayakta:
            if not self.kalkista:
                self.gecis += 1
            self.kalkista, self.sayac = True, 0
        elif self.kalkista:
            self.sayac += 1
            if self.sayac >= self.n_donus:
                self.kalkista = False
        ag = self.k if self.kalkista else self.d
        return ag.get_action(o, training=False)[0]


def olc(env, sec, N=24, aktif=True, tohum0=50_000):
    import torch
    isb, tmz, aln, ay, dev, kalk = [], [], [], [], 0, 0
    for i in range(N):
        np.random.seed(tohum0 + i); torch.manual_seed(tohum0 + i); sec.sifirla()
        o, _ = env.reset(options=dict(rakip_i=0, rakip_aktif=aktif, yerde=False)); done = False; n = a = 0
        onceki_ayakta, dustu_t = True, None
        while not done:
            ayk = env.P[0].ayakta_mi()
            iz = float(env.data.xmat[env.P[0].torso][6])
            o, _, te, tr, _ = env.step(sec.eylem(o, ayk, iz)); done = te or tr; n += 1; a += ayk
            if onceki_ayakta and not ayk:
                dev += 1
            if not onceki_ayakta and ayk:
                kalk += 1
            onceki_ayakta = ayk
        isb.append(env.isabet); tmz.append(env.isabet_temiz); aln.append(env.alinan); ay.append(a / n)
    return (f"{'AKTIF' if aktif else 'PASIF'} ({N}): isabet {np.mean(isb):.1f} (temiz {np.mean(tmz):.1f}) alınan {np.mean(aln):.1f} "
            f"| ayakta %{100 * np.mean(ay):.0f} | düşme {dev} → kalkma {kalk}")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.getcwd())
    import torch, selfplay_training as st
    st.DEVICE = torch.device("cpu"); st.USE_AMP = False; torch.set_num_threads(2)
    import alt_dovus_env as A
    yukle = lambda ck: (lambda a: (a.load(ck), a)[1])(st.CUDAPPOAgent(A.OBS_DIM, A.ACT_DIM, name="x"))
    dovus, kalkis = yukle(sys.argv[1]), yukle(sys.argv[2])
    env = A.AltDovusEnv(env_id=hash(sys.argv[2]) % 50 + 240)
    donme = None if len(sys.argv) <= 3 else DonmeHareketi(sys.argv[3], A.DT) if sys.argv[3].endswith(".npz") else yukle(sys.argv[3])
    sec = BeceriSecici(dovus, kalkis, A.DT, donme)
    ok = 0
    for s in range(12):                                  # sırtüstü kalkış (seçiciyle)
        np.random.seed(90000 + s); sec.sifirla(); o, _ = env.reset(options=dict(yerde=True, sirtustu=True, rakip_aktif=False))
        for n in range(int(8 / A.DT)):
            o, _, te, tr, _ = env.step(sec.eylem(o, env.P[0].ayakta_mi(), float(env.data.xmat[env.P[0].torso][6])))
            if env.P[0].ayakta_mi() and not sec.kalkista: ok += 1; break
            if te or tr: break
    hp = np.load(A.DUSUS_HAVUZU)["poz"]; Y = np.where(hp == "yuzustu")[0]; okY = 0
    for k in Y:                                          # gerçek YÜZÜSTÜ düşüşlerden tam zincir (dönme → kalkış), ≤ 8 s
        np.random.seed(73000 + int(k)); sec.sifirla(); o, _ = env.reset(options=dict(havuz=True, havuz_i=int(k)))
        for n in range(int(8 / A.DT)):
            o, _, te, tr, _ = env.step(sec.eylem(o, env.P[0].ayakta_mi(), float(env.data.xmat[env.P[0].torso][6])))
            if env.P[0].ayakta_mi(): okY += 1; break
            if te: break
    print(f"kalkış uzmanı {os.path.basename(sys.argv[2])}{' + dönme ' + os.path.basename(sys.argv[3]) if donme else ''}: "
          f"sırtüstünden kalkış {ok}/12 | YÜZÜSTÜ gerçek düşüşten kalkış {okY}/{len(Y)} | {olc(env, sec, 24, True)}", flush=True)
