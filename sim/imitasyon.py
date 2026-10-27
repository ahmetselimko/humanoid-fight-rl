# -*- coding: utf-8 -*-
"""TAKLİT (IMITATION) ODULU — referans kliplerini egitime baglama cekirdegi.

DeepMimic-lite: ajanin ANLIK pozunu bir referans poza kiyaslar, benzerligi
[0,1] odule cevirir. Iki kullanim:

  1) GetupTracker  — KALKMA icin. Ajan yerdeyken, kalkma referansini FAZ olarak
     takip eder (monoton ilerleme): hem referansa benzemeyi hem referans boyunca
     ILERLEMEYI (yatik -> dik) odullendirir. Kalkmanin SEKLINI ogretir.

  2) nearest_match — YURUME/DOVUS/DIK becerilerini SABITLEME (unutmayi onleme):
     ajanin pozu ilgili beceri klibindeki en yakin kareye ne kadar benziyor.

Referans format (referanslar/*.npz): qpos[T,19], qvel[T,18], dt.
  qpos: [root_xyz(3), root_quat(4, wxyz), 12 eklem]
Poz karsilastirmasi ROOT XY'den BAGIMSIZ: sadece z + govde-egimi + 12 eklem.
"""
import os, glob
import numpy as np

REF_DIR = "referanslar"

# ── poz ozellik cikarimi ───────────────────────────────────────────────────
def _tilt_from_quat(q):
    """Govde z-ekseninin dunya-z'den sapma acisi (0=dik, ~1.57=yatik)."""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    # body-up ekseninin dunya-z bileseni = R[2,2]
    zz = 1.0 - 2.0 * (x * x + y * y)
    zz = max(-1.0, min(1.0, zz))
    return float(np.arccos(zz))


def pose_features(qpos19):
    """qpos[19] -> (z, tilt, eklemler[12]) — konumdan bagimsiz poz imzasi."""
    q = np.asarray(qpos19, np.float32)
    z = float(q[2])
    tilt = _tilt_from_quat(q[3:7])
    joints = q[7:19].astype(np.float32)
    return z, tilt, joints


def imitation_reward(agent_qpos, ref_qpos,
                     w_joint=0.5, w_z=0.25, w_tilt=0.25,
                     k_joint=2.0, k_z=12.0, k_tilt=2.0):
    """Iki poz arasi benzerlik [0,1] (DeepMimic tarzi agirlikli us)."""
    za, ta, ja = pose_features(agent_qpos)
    zr, tr, jr = pose_features(ref_qpos)
    r_joint = np.exp(-k_joint * float(np.mean((ja - jr) ** 2)))
    r_z     = np.exp(-k_z * (za - zr) ** 2)
    r_tilt  = np.exp(-k_tilt * (ta - tr) ** 2)
    return float(w_joint * r_joint + w_z * r_z + w_tilt * r_tilt)


# ── referans kutuphanesi ────────────────────────────────────────────────────
class ReferenceLibrary:
    """referanslar/ altindaki tum klipleri yukler; getup referansini birlestirir."""

    def __init__(self, ref_dir=REF_DIR):
        self.clips = {}          # skill -> qpos[T,19]
        self._load(ref_dir)

    def _load(self, ref_dir):
        for f in sorted(glob.glob(os.path.join(ref_dir, "*.npz"))):
            name = os.path.splitext(os.path.basename(f))[0]
            d = np.load(f)
            self.clips[name] = d["qpos"].astype(np.float32)
        # hasat edilen kalkmalar (varsa) ayri klasorde
        for f in sorted(glob.glob(os.path.join(ref_dir, "getup_hasat", "*.npz"))):
            name = "hasat_" + os.path.splitext(os.path.basename(f))[0]
            d = np.load(f)
            self.clips[name] = d["qpos"].astype(np.float32)

    def getup_reference(self, on_back=True):
        """Kalkma referansi sec. DUSME-YONUNE gore (arkadan/onden) uretecten gelen
        referansi tercih et; yoksa elle-keyframe / hasat / en-uzun.
          on_back=True  -> sirtustu dustu  -> getup_arkadan
          on_back=False -> yuzustu dustu   -> getup_onden
        """
        # 1) uretec referanslari (yon-secimli)
        if on_back and "getup_arkadan" in self.clips:
            return self.clips["getup_arkadan"]
        if (not on_back) and "getup_onden" in self.clips:
            return self.clips["getup_onden"]
        # 2) diger yon uretec varsa onu da kabul et (tek yon uretildiyse)
        for k in ("getup_arkadan", "getup_onden"):
            if k in self.clips:
                return self.clips[k]
        # 3) elle-keyframe + hasat -> en uzun
        parts = []
        if "getup_keyframe" in self.clips:
            parts.append(self.clips["getup_keyframe"])
        for k, v in self.clips.items():
            if k.startswith("hasat_"):
                parts.append(v)
        if not parts:
            return None
        parts.sort(key=len, reverse=True)
        return parts[0]

    def has_getup(self):
        return self.getup_reference() is not None

    def summary(self):
        return {k: v.shape for k, v in self.clips.items()}


# ── kalkma faz-takipcisi (monoton) ──────────────────────────────────────────
class GetupTracker:
    """Ajanin kalkma referansi boyunca ILERLEMESINI takip eder.

    Her adim: ileri pencerede [idx, idx+W] en iyi eslesen kareyi bul, idx'i
    oraya ILERLET (geri gitmez). Odul = referansa-benzerlik * ilerleme-bonusu.
    reset() her episode/dusme basinda cagrilir.
    """

    def __init__(self, ref_qpos, window=40):
        self.ref = np.asarray(ref_qpos, np.float32) if ref_qpos is not None else None
        self.window = window
        self.T = len(self.ref) if self.ref is not None else 0
        self.idx = 0

    def reset(self):
        self.idx = 0

    def set_reference(self, ref_qpos):
        """Referansi degistir (dusme-yonune gore arkadan/onden) + fazi sifirla."""
        self.ref = np.asarray(ref_qpos, np.float32) if ref_qpos is not None else None
        self.T = len(self.ref) if self.ref is not None else 0
        self.idx = 0

    def available(self):
        return self.ref is not None and self.T > 1

    def step(self, agent_qpos):
        """-> (imit_reward[0..1], progress[0..1], advanced_frames). Yoksa (0,0,0)."""
        if not self.available():
            return 0.0, 0.0, 0
        hi = min(self.T - 1, self.idx + self.window)
        best_i, best_r = self.idx, -1.0
        for i in range(self.idx, hi + 1):
            r = imitation_reward(agent_qpos, self.ref[i])
            if r > best_r:
                best_r, best_i = r, i
        advanced = best_i - self.idx
        self.idx = best_i                       # monoton: sadece ileri
        progress = self.idx / (self.T - 1)
        return float(best_r), float(progress), int(advanced)


def getup_imitation_bonus(tracker, agent_qpos, w=5.0, shape_floor=0.3):
    """Kalkma taklit-BONUSU (kullanici formulu, 2026-08):
        bonus = w * benzerlik * ilerleme   (+ kucuk sekil-tabani ILERLERKEN)

      - benzerlik (imit) : anlik referans poza yakinlik [0,1]
      - ilerleme (progress): referans boyunca kat edilen yol [0,1] (yatik 0 -> dik 1)
        = "gorev-sinyali" (ne kadar yukseldin). VE-kapisi: full puan (~w=5) ancak
        HEM referansa benziyor HEM ilerleme yuksek (dike yakin) iken.
      - yatik-sabit:  ilerleme 0 -> bonus 0 (yatmaya odul YOK; el-gradyani -0.8 verir)
      - kalkma-yolunda: ilerleme buyur -> guclu pozitif (tepe ~w near dik)
      - daha iyi/farkli kalkma: referansa az benzese de gorev odulu (dik bonusu +1.0)
        baskin -> KAZANIR; sadece-kopya-ama-kotu-icra full alamaz (ilerleme dusuk).
      - sekil-tabani: yalniz ILERLERKEN kucuk ek (erken bootstrap; yatarken 0 kalir).
    -> (bonus, ilerleme)
    """
    imit, progress, advanced = tracker.step(agent_qpos)
    bonus = w * imit * progress                     # benzerlik x ilerleme
    if advanced > 0:                                # ilerlerken kucuk sekil-tabani
        bonus += shape_floor * imit
    return float(bonus), float(progress)


def nearest_match(agent_qpos, clip, stride=3):
    """Ajan pozunun klipteki en yakin kareye benzerligi [0,1] (unutma-onleme)."""
    if clip is None or len(clip) == 0:
        return 0.0
    best = 0.0
    for i in range(0, len(clip), stride):
        r = imitation_reward(agent_qpos, clip[i])
        if r > best:
            best = r
    return float(best)


if __name__ == "__main__":
    # ── standalone dogrulama ──
    lib = ReferenceLibrary()
    print("Yuklenen klipler:")
    for k, s in lib.summary().items():
        print(f"  {k}: {s}")

    ref = lib.getup_reference()
    if ref is None:
        print("\n!! Kalkma referansi yok (once keyframe_sablon.py calistir).")
    else:
        print(f"\nKalkma referansi: {ref.shape[0]} kare, "
              f"z {ref[0,2]:.2f} -> {ref[-1,2]:.2f}")
        tr = GetupTracker(ref)

        # TEST 1: referansin KENDISINI besle -> imit~1, progress 0->1
        tr.reset()
        rs, ps = [], []
        for i in range(len(ref)):
            r, p, _ = tr.step(ref[i])
            rs.append(r); ps.append(p)
        print(f"\nTEST1 referansi kendine besle: "
              f"imit_ort={np.mean(rs):.3f}  progress son={ps[-1]:.3f}")

        # TEST 2: YATIK pozda sabit kal (kalkmiyor) -> progress dusuk kalmali
        tr.reset()
        lying = ref[0].copy()
        rs2, ps2 = [], []
        for _ in range(len(ref)):
            r, p, _ = tr.step(lying)
            rs2.append(r); ps2.append(p)
        print(f"TEST2 yerde sabit kal:            "
              f"imit_ort={np.mean(rs2):.3f}  progress son={ps2[-1]:.3f}")

        # TEST 3: DIK pozda sabit (zaten ayakta) -> progress hizli 1'e
        tr.reset()
        standing = ref[-1].copy()
        rs3, ps3 = [], []
        for _ in range(len(ref)):
            r, p, _ = tr.step(standing)
            rs3.append(r); ps3.append(p)
        print(f"TEST3 dik sabit:                  "
              f"imit_ort={np.mean(rs3):.3f}  progress son={ps3[-1]:.3f}")

        # ── BONUS FORMULU testi: kalkma vs yatik vs firlama ──
        def run_bonus(feed_fn, label):
            tr.reset(); total = 0.0; lastp = 0.0
            for i in range(len(ref)):
                b, lastp = getup_imitation_bonus(tr, feed_fn(i))
                total += b
            return total, lastp
        tot_up, pr_up = run_bonus(lambda i: ref[i], "kalkma")
        tot_lie, pr_lie = run_bonus(lambda i: ref[0], "yatik")
        # firlama: yatik poz ama z cok yuksek + gurultu
        def launched(i):
            q = ref[0].copy(); q[2] = 2.0; return q
        tot_fl, pr_fl = run_bonus(launched, "firlama")
        print(f"\nBONUS toplam (episode boyu):")
        print(f"  kalkma(referansi izle): {tot_up:+.1f}  (progress {pr_up:.2f})")
        print(f"  yatik(yerde sabit):     {tot_lie:+.1f}  (progress {pr_lie:.2f})")
        print(f"  firlama(z=2 yatik):     {tot_fl:+.1f}  (progress {pr_fl:.2f})")

        # beceri klipleriyle nearest_match ornegi
        if "dik_durma" in lib.clips:
            std_pose = lib.clips["dik_durma"][len(lib.clips["dik_durma"])//2]
            print(f"\nnearest_match(dik poz, dik_durma klip) = "
                  f"{nearest_match(std_pose, lib.clips['dik_durma']):.3f}")
