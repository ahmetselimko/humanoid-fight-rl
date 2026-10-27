# -*- coding: utf-8 -*-
"""Dövüş kaydı (PLAN Faz 4, 2026-10-22) — kayıt biçimi: bölge zincirinde bir maç.
  set ROBOT=govde2 & set DOVUS_HIZ=1.5 & venv\\Scripts\\python.exe dovus_kayit.py [--taktik bulut_sonuc/taktik_ep80.pth]
      [--alt models/<hareket_agi>.pth] [--parca …_komsu3_fizik.xml] [--dunya arena] [--sure 40]
Kırmızı (P1) ve mavi (P2) = taktik ağ (açgözlü); --mavi kural → mavi kural tabanlı (eski ölçüt düzeni). Her ALT ADIMDA (DT ≈ 0.016 s → 62.5 kare/s) kayıt.
Çıktı: mekan/kayitlar/bolge_<tarih_saat>.npz
  fps, T · govde_adlari (B,) · konum (T,B,3) · quat (T,B,4) (w,x,y,z) · geom_govde/tip/boyut/konum/quat/renk (robot görünüşü)
  · mekan (fizik parçası yolu) · dunya · olaylar (isabet/devrilme zamanları, JSON) · ozet (JSON)
"""
import os, sys, json, time, argparse
os.chdir(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.getcwd())
import numpy as np, torch, mujoco
import selfplay_training as st
st.DEVICE = torch.device("cpu"); st.USE_AMP = False
import getup_train as G
import taktik_env as TK
from taktik_train import Ag

ap = argparse.ArgumentParser()
ap.add_argument("--taktik", default=os.environ.get("TAKTIK", "models/taktik3_son.pth"))   # 2026-10-27: en_iyi_sistem.sh
ap.add_argument("--alt", default=os.environ.get("ALT", "models/alt_dovus2_govde2_v3_epson3M.pth"))
ap.add_argument("--parca", default="arena/insa/oda_001/oda_001_komsu3_fizik.xml")
ap.add_argument("--dunya", default="arena")
ap.add_argument("--sure", type=float, default=40.0); ap.add_argument("--tohum", type=int, default=3)
ap.add_argument("--bolge", default="orman", help="maçın başladığı bölge türü")
ap.add_argument("--uzak", action="store_true", help="mavi KOMŞU bölgede başlar (kırmızı geçitten geçerek kovalar)")
ap.add_argument("--kalkis", default=os.environ.get("KALKIS", "models/alt_dovus2_govde2_kuzman3_epson3M.pth"), help="kalkış uzmanı (beceri seçici: düşünce o sürer)")
ap.add_argument("--donme", default=None, help="dönme uzmanı (yüzüstü → sırtüstü; beceri seçici)")
ap.add_argument("--gez", default=os.environ.get("GEZ", "models/alt_dovus2_govde2_gez_epson4M.pth"), help="rakip uzaktayken gezinme ağı ('' = kapalı)")
ap.add_argument("--mavi", choices=("ag", "kural"), default="ag",
                help="2026-10-24 kullanıcı kararı: robot-robot maçında mavi de taktik ağıyla (kural yalnız ölçüt)")
ap.add_argument("--canavar", action="store_true", help="canavar görevi (canavar_env): iki robot da taktik ağla canavara karşı")
ap.add_argument("--mavi-alt", default=None, help="2026-10-25 kafa kafaya: mavi başka hareket ağıyla (kırmızı --alt); "
                "adında 'gardref' geçen ağ gardlı referans ayarıyla vurur")
ap.add_argument("--mp4", default=None, help="2026-10-25: maçı MuJoCo görüntüleyicisiyle doğrudan videoya çek (Blender'dan hızlı)")
args = ap.parse_args()


def _ref_ayar(ck):
    """Ağın eğitildiği vuruş referansı: gardref → (kısa kesit 0.35/0.25 s, gardlı, geçiş 0.15 s); eskiler → (tam kesit, 0.3 s)."""
    return ((0.35, 0.25, True), 0.15) if "gardref" in os.path.basename(ck) else ((0.0, 0.0, False), 0.3)

TK.MAC_SURE_S = args.sure; TK.UZAK_BAS_P = 1.0 if args.uzak else 0.0
if args.canavar:
    import canavar_env as CE
    env = CE.CanavarEnv(args.parca, args.alt, args.dunya, env_id=120, bas_bolge=args.bolge)
else:
    env = TK.TaktikEnv(args.parca, args.alt, args.dunya, env_id=120)
if args.mavi_alt:
    (r1, g1), (r2, g2) = _ref_ayar(args.alt), _ref_ayar(args.mavi_alt)
    env.ikinci_alt_yukle(args.mavi_alt, r2, g2, r1, g1)
if args.kalkis:
    env.kalkis_yukle(args.kalkis)
if args.donme:
    env.donme_yukle(args.donme)
if args.gez:
    env.gez_yukle(args.gez)
_d = torch.load(args.taktik, weights_only=False)["ag"]; N_OBS = _d["govde.0.weight"].shape[1]   # 47 taktik, 51 canavar eğitimi
ag = Ag(N_OBS, TK.N_KOMUT); ag.load_state_dict(_d); ag.eval()
m, d = env.model, env.data
B = [b for b in range(m.nbody) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith(("p1_", "p2_", "canavar"))]
G_ = [g for g in range(m.ngeom) if m.geom_bodyid[g] in B]
TIP = {mujoco.mjtGeom.mjGEOM_CAPSULE: "capsule", mujoco.mjtGeom.mjGEOM_SPHERE: "sphere", mujoco.mjtGeom.mjGEOM_BOX: "box",
       mujoco.mjtGeom.mjGEOM_CYLINDER: "cylinder"}
konum, quat, olay = [], [], []
_asil = mujoco.mj_step; _say = [0]


_yazici = None
if args.mp4:
    import imageio_ffmpeg
    _rend = mujoco.Renderer(m, 360, 640); _kam = mujoco.MjvCamera(); _kam.type = mujoco.mjtCamera.mjCAMERA_FREE
    _kam.distance, _kam.elevation = 4.5, -22
    _t1, _t2 = (mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in ("p1_torso", "p2_torso"))
    _yazici = imageio_ffmpeg.write_frames(args.mp4, (640, 360), fps=31, quality=7, macro_block_size=8); _yazici.send(None)


def _kaydeden(mm, dd, *a, **k):
    _asil(mm, dd, *a, **k); _say[0] += 1
    if _say[0] % G.FRAME_SKIP == 0:
        konum.append(dd.xpos[B].copy()); quat.append(dd.xquat[B].copy())
    if _yazici is not None and _say[0] % (2 * G.FRAME_SKIP) == 0:          # ~31 kare/s
        orta = (dd.xpos[_t1] + dd.xpos[_t2]) / 2; fark = dd.xpos[_t2] - dd.xpos[_t1]
        _kam.lookat[:] = [orta[0], orta[1], 0.6]
        _kam.azimuth = float(np.degrees(np.arctan2(fark[1], fark[0]))) + 90.0     # iki robota yandan bak
        _rend.update_scene(dd, camera=_kam); _yazici.send(_rend.render().tobytes())


np.random.seed(args.tohum)
for _ in range(50):                                     # istenen bölgede başlat
    o = env.reset()
    if env.alan_turu[env.bolge(0)] == args.bolge:
        break
mujoco.mj_step = _kaydeden
try:
    bitti, onceki = False, dict(isabet=[0, 0], devrilme=[0, 0])
    t0 = time.time()
    while not bitti:
        with torch.no_grad():
            k1 = int(ag(torch.as_tensor(o[0][None, :N_OBS]))[0].argmax())
            k2 = (int(ag(torch.as_tensor(o[1][None, :N_OBS]))[0].argmax()) if args.canavar or args.mavi == "ag"
                  else TK.kural_komut(env, 1))
        o1, o2, _, _, bitti, ist = env.adim(k1, k2); o = (o1, o2)
        if args.canavar and ist.get("oldu"):
            olay.append(dict(t=round(env.t * TK.DT, 2), tur="canavar_oldu", kim="ikisi"))
        t = env.t * TK.DT
        for i, ad in ((0, "kirmizi"), (1, "mavi")):
            if ist["isabet"][i] > onceki["isabet"][i]:
                olay.append(dict(t=round(t, 2), tur="isabet", kim=ad))
            if ist["devrilme"][i] > onceki["devrilme"][i]:
                olay.append(dict(t=round(t, 2), tur="devrildi", kim=ad))
        onceki = dict(isabet=list(ist["isabet"]), devrilme=list(ist["devrilme"]))
finally:
    mujoco.mj_step = _asil
os.makedirs("mekan/kayitlar", exist_ok=True)
cikti = f"mekan/kayitlar/{'canavar' if args.canavar else 'bolge'}_{time.strftime('%Y%m%d_%H%M')}_t{args.tohum}.npz"
if _yazici is not None:
    _yazici.close(); print(f"video: {args.mp4}")
ozet = dict(sure_s=round(env.t * TK.DT, 1), isabet=list(map(int, ist["isabet"])), devrilme=list(map(int, ist["devrilme"])),
            bolge_degisimi=list(map(int, ist["bolge_degisimi"])), taktik=args.taktik, alt=args.alt,
            komutlar=dict(zip(TK.KOMUTLAR, map(int, ist["komut"][0]))), baslangic_bolge=args.bolge, kalkis=args.kalkis,
            **(dict(canavar_oldu=bool(ist["oldu"]), canavara_hasar=round(float(sum(ist["hasar"])), 1),
                    robotlara_hasar=round(float(sum(ist["alinan"])), 1)) if args.canavar else {}))
np.savez_compressed(
    cikti, fps=np.float32(1.0 / TK.DT), T=len(konum),
    govde_adlari=np.array([mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in B]),
    konum=np.array(konum, np.float32), quat=np.array(quat, np.float32),
    geom_govde=np.array([B.index(m.geom_bodyid[g]) for g in G_]), geom_tip=np.array([TIP.get(m.geom_type[g], "box") for g in G_]),
    geom_boyut=m.geom_size[G_].astype(np.float32), geom_konum=m.geom_pos[G_].astype(np.float32),
    geom_quat=m.geom_quat[G_].astype(np.float32), geom_renk=m.geom_rgba[G_].astype(np.float32),
    mekan=args.parca, dunya=args.dunya, olaylar=json.dumps(olay, ensure_ascii=False), ozet=json.dumps(ozet, ensure_ascii=False))
print(f"kaydedildi: {cikti} | {len(konum)} kare ({len(konum) * TK.DT:.1f} s, {time.time() - t0:.0f} s gerçek) | {ozet}")
