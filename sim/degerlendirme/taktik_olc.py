# -*- coding: utf-8 -*-
"""Taktik ağ sürümlerini KURALA karşı eşit koşullarda karşılaştır (2026-10-22): aynı tohumlar, açgözlü, beceri seçicili.
Bulut ölçümü 22 maçtı (sürümler arası ±%15 oynuyor) → burada sürüm başına N maç, aynı tohumlar.
  set ROBOT=govde2 & set KOL_KP=3 & set DOVUS_HIZ=1.5 & venv\Scripts\python.exe degerlendirme\taktik_olc.py N ck1 ck2 ...
"""
import os, sys
KOK = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); os.chdir(KOK); sys.path.insert(0, KOK)
import numpy as np
import multiprocessing as mp

# 2026-10-24: MEKAN_DUNYA=<dunya klasoru> → yerleşim modelinin ürettiği zincirde (fizik: insa/oda_001/oda_001_komsu3)
DUNYA = os.environ.get("MEKAN_DUNYA", "arena"); PARCA = f"{DUNYA}/insa/oda_001/oda_001_komsu3_fizik.xml"
HAVUZ = int(os.environ.get("HAVUZ", "4"))   # yerelde ısınma: 3 yeter
ALT = os.environ.get("ALT", "models/alt_dovus2_govde2_v3_epson3M.pth"); KALKIS = os.environ.get("KALKIS", "models/alt_dovus2_govde2_kuzman3_epson3M.pth")
# 2026-10-23: GEZ=<ağ.pth> verilirse TAKTİK oyuncusu rakip uzaktayken gezinme ağını kullanır (GEZ_ESIK="uzak,yakın" m)
BAS_BOLGE = os.environ.get("BAS_BOLGE")   # ör. orman: maç o bölgede başlar (engebe ölçümü, ENGEBE_CM ile)
GEZ = os.environ.get("GEZ"); GEZ_ESIK = [float(x) for x in os.environ.get("GEZ_ESIK", "2.5,1.8").split(",")]


def mac(args):
    ck, tohum = args
    global _E
    import torch, random
    torch.set_num_threads(1)
    import selfplay_training as st
    st.DEVICE = torch.device("cpu"); st.USE_AMP = False
    import taktik_env as TK, taktik_train as TT
    if "_E" not in globals():
        _E = TK.TaktikEnv(PARCA, ALT, DUNYA, env_id=400 + os.getpid() % 50); _E.kalkis_yukle(KALKIS)
        if GEZ: _E.gez_yukle(GEZ, *GEZ_ESIK)
    ag = TT.Ag(TK.OBS, TK.N_KOMUT); ag.load_state_dict(torch.load(ck, weights_only=False)["ag"])
    np.random.seed(tohum); random.seed(tohum); torch.manual_seed(tohum)
    i = tohum % 2; o = _E.reset(); bitti = False
    while BAS_BOLGE and _E.alan_turu[_E.bolge(0)] != BAS_BOLGE:
        o = _E.reset()
    if GEZ: _E.gez_oyuncular = {i}
    _E.vurus_fren = {i}                  # VURUS_FREN_HIZ verilirse yalnız taktik oyuncusu frenler
    while not bitti:
        a = TT._sec(ag, o[i], True)[0]; b = TK.kural_komut(_E, 1 - i)
        o1, o2, _, _, bitti, ist = _E.adim(*((a, b) if i == 0 else (b, a))); o = (o1, o2)
    return (TT.skor(ist, i), ist["isabet"][i], ist["isabet"][1 - i], ist["bolge_degisimi"][0] + ist["bolge_degisimi"][1],
            ist["gez"][i] / max(1, ist["ayakta"][i]), ist["devrilme"][i])


if __name__ == "__main__":
    N = int(sys.argv[1])
    with mp.get_context("spawn").Pool(HAVUZ) as h:
        for ck in sys.argv[2:]:
            R = np.array(h.map(mac, [(ck, 5000 + t) for t in range(N)]))
            print(f"{os.path.basename(ck):18s} ({N} maç): kazanma %{100 * np.mean(R[:, 0] > 0):.0f} | skor ort {R[:, 0].mean():+.1f} "
                  f"| isabet {R[:, 1].mean():.1f} alınan {R[:, 2].mean():.1f} | bölge değişimi/maç {R[:, 3].mean():.2f} "
                  f"| gez ağı %{100 * R[:, 4].mean():.0f} | düşme/maç {R[:, 5].mean():.1f}", flush=True)
