# -*- coding: utf-8 -*-
"""Engebeli zemin (Faz 5, 2026-10-23): sahne XML'ine yükseklik alanı (hfield) ekler; düz zemin çarpışmadan çıkar.
Tümsekler yalnız verilen dikdörtgenlerde (ör. orman bölgesi), kenarlara doğru 1 m'de düze iner; dışarısı z = 0.
Ölçüldü (degerlendirme/engebe_olcum.py, engebe_dovus_olcum.py): tepe-çukur ≤ 12 cm'de yürüme/kalkış eğitimsiz çalışıyor.
  xml_ekle(xml, genlik) → bilgi;  model kurulduktan sonra doldur(model, bilgi, kutular, tohum);  zemin_z(model, x, y)
"""
import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter

HUCRE, OLCEK_M, KENAR_M = 0.1, 0.5, 1.0


def xml_ekle(xml, genlik, pay=3.0, kutu=None):
    """xml dosyasını yerinde değiştirir. hfield bütün mekân geomlarını (pay kadar fazlasıyla) ya da verilen kutuyu
    ((xmin, ymin), (xmax, ymax)) kaplar. Düz zemin 0.3 m aşağı iner (hfield dışına çıkılırsa yedek; tümseğe değmez)."""
    if kutu is None:
        m = mujoco.MjModel.from_xml_path(xml); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
        xy = np.array([d.geom_xpos[g][:2] for g in range(m.ngeom)
                       if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("mekan_")])
        kutu = (xy.min(0) - pay, xy.max(0) + pay)
    lo, hi = np.asarray(kutu[0], float), np.asarray(kutu[1], float); c, yari = (lo + hi) / 2, (hi - lo) / 2
    nr, nc = int(2 * yari[1] / HUCRE) + 1, int(2 * yari[0] / HUCRE) + 1
    s = open(xml, encoding="utf-8").read()
    s = s.replace('<geom name="floor" ', '<geom name="floor" pos="0 0 -0.3" ', 1)
    i = s.index("</asset>")
    s = s[:i] + f'<hfield name="engebe" nrow="{nr}" ncol="{nc}" size="{yari[0]:.3f} {yari[1]:.3f} {genlik} 0.1"/>' + s[i:]
    i = s.rindex("</worldbody>")
    s = s[:i] + (f'<geom name="engebe" type="hfield" hfield="engebe" pos="{c[0]:.3f} {c[1]:.3f} {-genlik / 2:.4f}" '
                 f'rgba="0.42 0.36 0.25 1"/>') + s[i:]
    open(xml, "w", encoding="utf-8").write(s)
    return dict(genlik=genlik, merkez=c, yari=yari, nr=nr, nc=nc)


def gurultu(bilgi, tohum=0):
    """[-0.5, 0.5] normalize yumuşak gürültü (±2σ = ±0.5), bilgi ızgarasında."""
    h = gaussian_filter(np.random.default_rng(tohum).standard_normal((bilgi["nr"], bilgi["nc"])), OLCEK_M / HUCRE)
    return np.clip((h - h.mean()) / (4 * h.std() + 1e-9), -0.5, 0.5)


def doldur(model, bilgi, kutular, tohum=0):
    """kutular: [(xmin, ymin, xmax, ymax)] — tümsekli alanlar. ±2σ = ±genlik/2 (engebe_olcum.arazi ile aynı)."""
    nr, nc, c, yari = bilgi["nr"], bilgi["nc"], bilgi["merkez"], bilgi["yari"]
    h = gurultu(bilgi, tohum)
    X, Y = np.meshgrid(c[0] + np.linspace(-yari[0], yari[0], nc), c[1] + np.linspace(-yari[1], yari[1], nr))
    w = np.zeros((nr, nc))
    for x0, y0, x1, y1 in kutular:
        ic = np.minimum.reduce([X - x0, x1 - X, Y - y0, y1 - Y])          # kutu kenarına içten uzaklık
        w = np.maximum(w, np.clip(ic / KENAR_M, 0, 1))
    model.hfield_data[:] = (0.5 + h * w).ravel()


def zemin_z(model, x, y):
    g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "engebe")
    if g < 0:
        return 0.0
    nr, nc = model.hfield_nrow[0], model.hfield_ncol[0]; gx, gy, gz = model.geom_pos[g]; sx, sy, sz, _ = model.hfield_size[0]
    i = int(np.clip(round((y - gy + sy) / (2 * sy) * (nr - 1)), 0, nr - 1))
    j = int(np.clip(round((x - gx + sx) / (2 * sx) * (nc - 1)), 0, nc - 1))
    return float(gz + model.hfield_data[i * nc + j] * sz)
