# -*- coding: utf-8 -*-
"""AYAĞA-KALKMA (GETUP) — AYRI, TEMİZ EĞİTİM.

Kullanici hakli olarak itiraz etti: getup 'ayri egitim' diye sunuldu ama aslinda
selfplay_training.py'nin dovus-makinesi icine GETUP_PHASE bayragiyla gomulmustu
(dovus odulleri, HP, knockdown sonlandirmasi ayni dosyada dallaniyordu).
Bu dosya getup'i GERCEKTEN ayirir:

  - KENDI ortami   : GetupEnv — kendi reset (dik+itme), kendi odul, kendi
                     sonlandirma (SADECE timeout). Dovus odulu/HP/knockdown YOK.
  - KENDI dongusu  : kendi PPO rollout + GAE + update dongusu.
  - AYNI XML/gozlem: dual_humanoid_fight.xml, 95-boyut gozlem KORUNUR
                     -> checkpoint dovus ajanina yuklenebilir, yuruyucuden
                        warm-start yapilabilir. (Yeni XML = obs boyutu degisir
                        = model uyumsuz olur; bu yuzden AYNI XML sart.)

MUFREDAT (yeni): itme kucuk baslar (ajan ayakta kalabilir -> POZITIF sinyal alir),
basari orani yukseldikce itme buyur (dusme -> kalkma pratigi). Seyrek-odul
duvarini kirmak icin: once 'dengede kal', sonra 'dusersen kalk'.
"""
import os, sys, glob, time
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import mujoco
import torch

import selfplay_training as st

# Faz bayraklarini KAPALI tut — bu dosya onlara BAGLI DEGIL (kendi mantigi var).
st.LOCOMOTION_PHASE = False
st.GETUP_PHASE      = False
st.COMBINE_PHASE    = False
st.COMBAT_VS_FROZEN = False
st.USE_IMITATION    = False

# ROBOT (2026-10-13): bilek MOTORLU + 20 cm ayak (dual_humanoid_getup_motor.xml).
# Orijinal robotta (8 cm ayak, motorsuz bilek) odul/gurultu/itme/KL duzeltmelerinin
# hicbiri comelme->ayakta gecisini acmadi; PD/referans/optimize de kalkamadi.
# Aksiyon 10->12 (gozlem bilekleri zaten iceriyordu) -> eski checkpoint'lerle UYUMSUZ.
XML          = "dual_humanoid_getup_motor.xml"
# ROBOT=kalcaz (2026-10-18, onaylı plan): kalça YAW eklemli robot (dual_humanoid_getup_motor_kalcaz.xml; motorlar 12 → 14,
# yeni motorlar listenin sonunda). Ayarlanmazsa eski robot BİREBİR aynı çalışır.
ROBOT = os.environ.get("ROBOT", "")
if ROBOT == "kalcaz":
    XML = "dual_humanoid_getup_motor_kalcaz.xml"
# ROBOT=govde2 (2026-10-21, PLAN_EKLEMLER.md): kalça yaw + BEL (p*_bel_z; üst gövde = göğüs+baş+kollar) + OMUZ YANA AÇMA
# (p*_shoulder_x_*). Motorlar 12 → 17 (yeniler sonda). XML degerlendirme/govde2_xml_uret.py ile üretilir.
if ROBOT == "govde2":
    XML = "dual_humanoid_getup_motor_govde2.xml"
# ESKİ DÜZEN (19 qpos / 18 qvel, 12 eklem): referanslar, havuzlar, taklit ödülü bu düzende. Yeni robotta eklemler İSİMLE eşlenir,
# eski düzende olmayan eklemler (kalça yaw) yazılırken 0'lanır, okunurken atlanır.
ESKI_EKLEMLER = ("shoulder_right", "elbow_right", "shoulder_left", "elbow_left", "hip_x_left", "hip_y_left", "knee_left",
                 "ankle_left", "hip_x_right", "hip_y_right", "knee_right", "ankle_right")
ESKI_XML     = "dual_humanoid_fight.xml"   # motorsuz bilek (k=30 -> calisma aninda 120)
# EPISODE SURESI — KOK BULGU (2026-08): fizik dt=0.003 s, yani 400 adim = SADECE
# 1.20 saniye. Dusme ~0.43 s aliyor -> kalkmaya 0.77 s kaliyor. Bizim kendi
# kalkma referansimiz 1.40-1.65 s suruyor: ajan kalkmayi TAMAMLAYAMIYOR, sure
# doluyordu. Tum %0 sonuclarin muhtemel gercek sebebi buydu (seyrek odul degil).
# FIZIK COZUNURLUGU (2026-08, kullanici: "yuksek gercekcilik aramiyorum"):
# XML'deki dt=0.003/iter=50 yerine dt=0.008/iter=10. Olculdu: ayni us/adim
# maliyetle 3.7x daha cok SIMULE ZAMAN (52x -> 192x gercek-zaman), ve 8/8
# kararlilik testinden temiz gecti (patlama yok, davranis ayni).
# NOT: XML DOSYASI DEGISTIRILMIYOR — sadece bu ortamda calisma aninda ayarlaniyor,
# boylece dovus egitimi ve 95-boyut gozlem uyumlulugu bozulmaz.
PHYS_DT      = 0.008        # fizik adimi (XML'deki 0.003 yerine)
SOLVER_ITER  = 10           # cozucu iterasyonu (XML'deki 50 yerine)
FRAME_SKIP   = 2            # 1 aksiyon = 2 fizik adimi -> 0.016 s = 62 Hz kontrol
# ── PD (HEDEF ACI) EYLEM UZAYI (2026-10-17, kullanici onayi) ─────────────────────────
# Ders 11 (degerlendirme/diz_olu_bolge.py): tork kontrolunde sinirdaki diz IKI DURUMLU — komut >=0 -> duz
# (olu bolge), -0.2 -> 1 s'de tamamen katlanip sirtustu dusme; ara aci yok, politika 'diz bük'u hic
# ogrenemedi. Fizibilite (degerlendirme/pd_fizibilite.py): PD'de gurultu oturusu bozmuyor ve dizi kucuk
# acilarla bukuyor, ara acilar kararli (-1.00/-1.50 tam), fizik aciliminin PD hedefleri robotun kendi
# oturusundan ACIK DONGU 1/2 kalkiyor. Eylem a in [-1,1] -> hedef = orta + a*PD_ASIM*yari_aralik (sinira
# kirpilir); tork = clip((PD_KP*(hedef-q) - PD_KD*qd)/gear) HER FIZIK ADIMINDA (fizik acilimini ureten
# video_fizik_opt.py ile ayni). PD_KONTROL=False: eski tork davranisi (eski tork checkpoint'leri icin).
PD_KONTROL   = True
PD_KP, PD_KD = 300.0, 30.0
# 2026-10-25 (kullanıcı: fren terimini adım adım azalt): PD_KD_K ile ölçeklenir (varsayılan 1 = eski davranış)
PD_KD *= float(os.environ.get("PD_KD_K", "1"))
# KOL_KP (2026-10-22, ölçüldü): KP 300 ile elin ulaşabildiği en yüksek hız 1.63 m/s (motor tam güçte 6.0, KP×3 ile 3.4) —
# yumruk yavaş görünüyordu. Kol motorlarında (omuz, dirsek; omuz-x dahil) KP × KOL_KP, KD × √KOL_KP. Varsayılan 1 = eski davranış.
KOL_KP = float(os.environ.get("KOL_KP", "1"))


def pd_kazanc(eklem_adlari):
    """Eklem adlarına göre (KP, KD) vektörleri: kol eklemleri KOL_KP ile sertleşir, diğerleri PD_KP/PD_KD."""
    kol = np.array([("shoulder" in a or "elbow" in a) for a in eklem_adlari])
    return np.where(kol, PD_KP * KOL_KP, PD_KP), np.where(kol, PD_KD * np.sqrt(KOL_KP), PD_KD)
PD_ASIM      = 1.1        # |a|~0.9'da sinira ulasir (ayakta diz ~sinirda: doygunluk olmasin)
# ARTIK (RESIDUAL) EYLEM (2026-10-17, kullanici: 'hangisini oneriyorsan'): mutlak eslemede eylem 0 = aralik
# ortasi = derin comelme -> sifirdan PD egitimi 1.2M adimda hic ayakta kalamadi (politika ~0 uretip comeliyor).
# 'Eylem 0 = ayakta' da cozum degil: yatan/oturan robota 'dizler duz' der (tork donemindeki yanlilik).
# Artik: hedef = TABAN + a * ARTIK_PAY. TABAN = referans hareketin o anki PD hedefi (yerde/zincir: fizik
# acilimini URETEN hedef dizisi + bilek dengesi, faz takibiyle; rsi/oturma/toplama: oturma ref pozu;
# itme/referanssiz: ayakta). Eylem 0 = 'hareketi aynen uygula'; politika yalniz duzeltme ogrenir.
EYLEM_MODU   = "artik"     # "artik" | "mutlak" (orta + a*PD_ASIM*yari)
ARTIK_PAY    = 1.0         # rad: |a|=1 -> tabandan 1 rad sapma. pd_artik_test.py (sigma 0.08): yerde 5/12 (0.5 ile 2/12)
TABAN_ONDE   = 1           # ref hedef dizisinde faz + bu kadar kare (hedefler[k] -> durum Q[k] uretir)
TABAN_BILEK  = True        # tum tabanlarda bilek denge geri beslemesi (govde dikken; aciliimi ureten denetleyici)
TUT_MAX_S    = 0.5         # artik: 'once topla' beklemesi en fazla bu kadar (sinirsiz bekleme tabani donduruyordu).
                           # 6M modeli: 0.5 s -> oturma 36/36, toplama 31/36; 1.0 s -> 32/36, 31/36
TABAN_KC     = 1.2         # yerde/zincir disi tabanlarda bilek kazanci (171441 aciliminin kc'si)


def ckpt_kontrol_modu(ck):
    """Checkpoint adina gore eylem uzayi: getup_artik_* -> PD artik, getup_pd_* -> PD mutlak, digerleri -> tork. Degerlendirme
    betikleri modeli yukledikten sonra cagirir; yanlis modda eski model anlamsiz davranir."""
    global PD_KONTROL, EYLEM_MODU
    ad = os.path.basename(str(ck))
    # "artik" adin herhangi bir yerinde (2026-10-17 hata: KALKIS_MODELI_artik_6M.pth / YEDEK_getup_artik_* yalniz
    # 'getup_artik' ON EKIYLE aranirken TORK moduna dusuyordu -> kalkis modeli 0/6 gorundu, model saglamdi)
    PD_KONTROL = ad.startswith("getup_pd") or "artik" in ad
    EYLEM_MODU = "mutlak" if ad.startswith("getup_pd") else "artik"
    return PD_KONTROL
MAX_STEPS    = 200          # 200 karar x 2 x 0.008 = 3.2 saniye
# AYAK BILEGI (2026-10, olculdu): ayak bilegi MOTORSUZ, pasif yay k=30 kutle
# merkezini tasiyamiyor -> eklemler sabit tutulsa bile govde itmesiz ~1.8 s'de
# one devriliyor (0/6). k=120 ile sabit poz itmesiz 6/6 kez 6 s ayakta kaliyor.
# Yani k=30'da 'ayakta durmak' fiziksel olarak cok zordu; politika once bunu
# asamadan kalkmayi ogrenemezdi. XML'e dokunulmaz (calisma aninda ayarlanir).
ANKLE_STIFFNESS = 120.0
NUM_ENVS     = 8            # paralel ortam
CKPT_PREFIX  = "getup_serial_k120_ep"   # 2026-10: eski getup_solo_* ESKI FIZIK (bilek k=30)
SAVE_FREQ    = 25
WARMUP_UPD   = 15           # critic kalibrasyonu (actor donuk)

# ── odul sabitleri (feet-gate: pozitif SADECE ayaklar yerdeyken) ──────────
FOOT_PLANTED_Z = 0.18
CALM_VZ        = 2.5
# GOVDE UZATMA (2026-10-14): XML'de kalca baglantisi 9 cm asagi (-0.16 -> -0.25), bas 0.75 kg.
# Tum yukseklik esikleri bu kadar kayar. Video/RSI referanslari ESKI govdeyle uretildi ->
# taklitte referans kok z'ye GOVDE_EK*cos(egim) eklenir (_pose_feats ref=True).
GOVDE_EK       = 0.09
STAND_Z        = 0.60 + GOVDE_EK   # 'ayakta' sayilma esigi (degerlendirme icin)
STAND_TORSO    = 0.70
STAND_HEAD_Z   = 0.79 + GOVDE_EK   # tam dik durustaki bas yuksekligi (eski govdede olculdu 0.79)
STAND_BONUS    = 2.0        # gercek ayakta durus (STAND_Z + STAND_TORSO) ek odulu

# ── itme siddeti: RAMPA YOK (kullanici tespiti) ──────────────────────────
# ESKI (hatali): itme kucuk baslar, ajan AYAKTA KALMAYI basarinca buyurdu.
#   Sonuc: ajan cogu episode hic DUSMUYORDU (dusen=1-4/8) -> kalkma PRATIGI
#   yapamiyordu. Mufredat, ajani hedef-disi ('ayakta kal') optimize etmeye itti.
# YENI: her episode GENIS araliktan rastgele itme. Boylece her batch'te hem
#   dusme (kalkma pratigi) hem kolay (pozitif sinyal) ornekleri bulunur.
# 2026-10: itme artik DENGE ogretir (dusme/kalkma pratigini yerde+RSI saglar).
# 0.5-2.0'da sifir aksiyon ~0.6 s'de dusuyordu -> 'ayakta kal' sinyali yoktu.
# 2026-10 (olculdu, 10 deneme ort. getiri): itme 0 -> PD-ayakta 1523 > comelme 1011;
# itme 0.15 -> PD 531 < comelme 827; 0.3 -> 368 < 867. Yani en kucuk itmede bile
# COMELME rasyonel optimumdu (itmeye karsi ayakta kalmak adim atma ister) ->
# politika ayakta baslayip comeliyordu. Hedef kalkma; itmeye dayaniklilik
# kalkma ogrenildikten SONRA ayri mufredat olarak eklenebilir.
PUSH_MIN     = 0.0          # itmesiz = saf denge
# 2026-10-13 motorlu bilek: bilek geri beslemeli PD itme 0.3'te 6/6, +gurultu 0.08 ile 6/6
# ayakta kaliyor (eski robot 0.15'te bile dusuyordu) -> 0.3'e cikarildi.
PUSH_MAX     = 0.3          # eski robotta 0.1 (ondan once 1.0, 2.0)

# ── BASLANGIC DAGILIMI (2026-10) ─────────────────────────────────────────
# Sadece itme ile baslayinca 500 update / ~2000 dususte TEK kalkma girisimi
# olmadi: politika 'dusmeyi geciktir' yerel optimumuna oturdu (yerde gecen
# sure ogrenilecek bir sey sunmuyordu). Cozum:
#   - YERDE baslangic : kalkma pratigi HER batch'te dogrudan yapilir.
#   - RSI (Reference State Init, DeepMimic): referans kalkma yorungesinin
#     rastgele bir karesinden basla -> ajan 'yari kalkmis' durumlari ve
#     yuksek odullerini gorur; uzun kredi-atama zinciri parcalara bolunur.
START_PROBS  = dict(itme=0.15, yerde=0.20, rsi=0.10, oturma=0.20, toplama=0.15, zincir=0.20)
# zincir (2026-10-14): fizik-dogrulanmis tam yerden kalkis acilimlarinin (YERDE_VIDEO_FILES)
# rastgele ara karesinden, GERCEK hizlariyla baslar; ayni acilim taklit hedefi. Zincirin her
# asamasi (dogrulma, diz cekme, one egilme, kalkis) ayri ayri pratik edilir.
# TOPLAMA (2026-10-13): mimic3 8M'de diz cekili oturmadan kalkis ogrenildi (kuvvetsiz 6/12)
# ama ajan kendi uzun oturmasindan bacaklarini hic toplamiyor (TOPLADI 0/93). Kullanicinin
# tam zincir videolarindaki (1714xx) 'uzun oturma -> diz cekme' kesitinden fizikte oturtulmus
# ara durumlar (degerlendirme/toplama_havuzu_uret.py): toplamayi kucuk parcalara boler.
TOPLAMA_FILE = "referanslar/toplama_havuzu.npz"
# ── 2026-10-13 OTURMA PLATOSU PLANI (kullanici + yayp-bilgisayar uzlasisi) ──
# Motorlu robot 10M adimda: itme 12/12 ayakta, ama yerde 10-11/12 OTURUP kaliyor.
# (HoST 2025, HumanUP 2025, Tao vd. 2022'den uyarlanan) uc ek:
#  1) YARDIM KUVVETI: govde dik (>ASSIST_TORSO) ve bas alcakken (<ASSIST_HEAD_Z)
#     torso'ya yukari kuvvet (assist_force, N). Egitim scripti basariya gore
#     kademe kademe 0'a indirir. VARSAYILAN 0 -> olcum/izleme her zaman kuvvetsiz.
#  2) Potansiyel tabanli bas-yuksekligi terimi: W*(gamma*Phi(s')-Phi(s)), Phi=bas z
#     orani. Optimum politikayi degistirmez (Ng 1999), ara pozlari odullendirir.
#  3) 'oturma' baslangici: ajanin KENDI oturma durumlarindan (havuz dosyasi,
#     egitim scripti her 2M adimda yeniler -> ters mufredat).
ASSIST_TORSO   = 0.6       # govde dikligi esigi (oturma/comelme/diz cokme)
# 2026-10-13 (10M degerlendirmesi, dovus-5b + yayp-bilgisayar ortak onerisi, kullanici onayi):
# bas<0.6'da sabit kuvvet 'protez' gibi calisti — kuvvetliyken kalkiyor, kuvvetsizken
# 11/12 oturuyor, kuvvetsiz kalkma 5 olcumde 0. Artik kuvvet YALNIZ kalkisin ilk itisinde:
# bas <= ASSIST_HEAD_FULL tam kuvvet, ASSIST_HEAD_FULL..ASSIST_HEAD_Z dogrusal azalir,
# ustu 0 -> 'yari kalkmis -> ayakta' kismini politika kendisi yapmak zorunda.
ASSIST_HEAD_FULL = 0.38 + GOVDE_EK   # kuvvetsiz oturmada bas z 0.36-0.38 (eski govde)
ASSIST_HEAD_Z  = 0.50 + GOVDE_EK     # bu yukseklikte ve ustunde kuvvet 0
POTENTIAL_W    = 5.0       # ayakta bonusu (2.0) / ayakta odulu (7.8) yaninda kucuk
POTENTIAL_GAMMA = 0.99     # PPO gamma ile AYNI (yayp: aksi halde ayakta sabit dururken sapma)
HAVUZ_FILE     = "oturma_havuzu.npz"   # qpos[N,19], qvel[N,18] (yalniz P1)
HAVUZ_QPOS_NOISE = 0.02    # rad, eklem gurultusu
HAVUZ_QVEL_NOISE = 0.05
# Eski kinematik refler (score=0, fizikte takip edilemiyor) -> YALNIZ yerde yatma pozu (ilk kare).
REF_FILES    = ("referanslar/getup_arkadan.npz", "referanslar/getup_onden.npz")
# RSI (2026-10-13): kullanicinin kendi kalkma videolarindan, FIZIKTE DOGRULANMIS acilimlar
# (yayp-bilgisayar: MediaPipe -> sagital referans -> fizik optimizasyonu; dovus-5b bagimsiz
# dogrulama). 3 tanesi kalca USTU oturmadan (bas z 0.37 = politikanin takildigi poz;
# kalca-150 robotunda uretildi), 2 tanesi ayak ustu comelmeden kalkis.
# v2 (2026-10-13): ilk video refslerinde govde egimi ISARETI TERSTI (insan one egilirken
# robot geriye yatiyordu; yayp buldu). Duzeltilmis v2 acilimlari, bagimsiz dogrulama
# (anahtar-poz PD kontrolcusu, baslangic gurultusu 0/0.02/0.05): 144607 11/11, 144638
# 11/11 ayakta. Eski 5 ref (yanlis isaret) cikarildi. Ikisi de diz cekili OTURMADAN kalkis.
RSI_FILES    = ("referanslar/video_144607_v2_oturma.npz",
                "referanslar/video_144638_v2_oturma.npz")
OTURMA_REF_FILES = RSI_FILES   # taklit eslesmesi + diz-cekili oturma karisimi (hepsi oturmadan)
RSI_MAX_FRAC = 0.85         # son %15 (zaten dik) kareler RSI'da kullanilmaz
JOINT_NOISE  = 0.05         # rad, eklem gurultusu (yerde/rsi)
SETTLE_STEPS = 25           # yerde baslangicta sifir-aksiyonla oturma (0.2 s)
# ── TAKLIT ODULU (2026-10-13, k150 4M degerlendirmesi sonrasi) ──────────────
# Kalca-150'ye ragmen politika kalcayi ~145 derece bukup ONE KATLANIP oturuyor; video
# durumlarindan basladiginda bile bitiremiyor (rsi 0/12). Eksik olan koordinasyon:
# 'one egil, agirligi ayaklara al, diz+kalcayi birlikte ac'. Faz bazli taklit:
#   r_im = MIMIC_W * [0.5*exp(-2*sum(eklem-ref)^2) + 0.3*exp(-40*(z_kok-ref)^2)
#                     + 0.2*exp(-10*(egim-ref)^2)]
# Hedef = video ACILIMININ qpos'u (ulasilabilir durumlar; PD ayar noktalari degil).
# Bilekler disarida (acilimdaki bilek, denge geri beslemesinin urunu — ajanin kendi
# denge stratejisini kilitlemeyelim). Aktif: rsi (secilen ref/kare) ve oturma
# (k150 reflerin ilk %30'unda en yakin kare; esik ustunde KAPALI).
MIMIC_W        = 2.0
MIMIC_EKLEM_K  = 0.5       # eklem basina cekirdek genisligi (exp(-K*fark^2))
# FAZ TAKIBI (2026-10-14, kullanici onayi): yerde/zincir fizik-acilim taklidinde hedef SAATE bagliydi.
# Olculdu (degerlendirme/diz_odul_teshis.py): kalkamayan robot icin diz cekmek yalniz hedefin 1.6-3.6 s
# penceresinde karli (+0.75/adim), 3.6-10 s'de hedef 'ayakta, dizler duz' oldugu icin zararli (-0.45/adim)
# -> duz bacakla oturmak daha cok odul (5.21 vs 5.02/adim): odul dizi DUZ tutmayi ogretiyordu.
# Artik faz = robotun pozuna en yakin ref karesi (yalniz ileri, adim basina en fazla FAZ_MAX_ILERI kare),
# taklit hedefi = faz + FAZ_ONDE kare. Duz oturan robotun fazi ~1.36 s'de durur, hedef diz cekme karesinde
# BEKLER; diz cektikce faz ve hedef kalkisa ilerler.
FAZ_MAX_ILERI = 8          # kare/kontrol adimi (ref dt 0.008: gercek zamanin 4 kati)
FAZ_PENCERE   = 250        # en yakin kare ~2 s ileriye kadar aranir (dar pencere yerel minimumda takiliyordu:
                           # hizli oturan robotun fazi 0.19 s'de kaldi); faz yine en fazla FAZ_MAX_ILERI ilerler
FAZ_ONDE      = 31         # hedef, fazin ~0.25 s onunde (0.5 s: cekili oturus kok-z payini 1.43 -> 0.46 dusuruyordu)
MIMIC_JOINTS   = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10]   # 12 eklemden bilekler (7, 11) haric
MIMIC_REF_STEP = 2         # ref dt 0.008 (5 dosyada dogrulandi) / kontrol dt 0.016
# ── YERDEN KALKIS: VIDEO TAKLIDI (2026-10-14, kullanici onayi — 'yalniz B') ──────────
# Insan-robot karsilastirmasi (degerlendirme/insan_robot_karsilastir.py) gosterdi: kullanicinin
# gercek sirtustu pozundan robot oturmaya bile ulasamiyor (bacaklari havaya savurup geriye
# yuvarlaniyor); insan ise govdeyi one kivirip oturuyor, diz cekiyor, +51 derece ONE egilip
# 3.9 s'de ayakta. 'yerde' bolumlerinde taklit hedefi HIC yoktu ve bolum 3.2 s idi.
# -> 'yerde' bolumlerinin YERDE_VIDEO_FRAC'i videonun ilk karesinden baslar ve AYNI videonun
#    tam zinciri zaman eslemeli taklit hedefi olur; 'yerde' bolumleri YERDE_SURE_S.
#    Kalan kisim eski yuzustu/sirtustu yatis pozlari (yuzustu kalkis videosu yok).
# (2026-10-14 kullanici tercihi) YALNIZ SON GRUP videolari: agirligi ONE vererek dusmeyi
# engelliyor (kalkis oncesi kollar arkada destekte %6-22). Ilk gruplar (114319, 141153)
# kollarla ARKADAN destekleniyor (%75-92) -> kullanilmiyor. 171457/171509/171540 tam zincir degil.
# 2026-10-14: KINEMATIK videolar yerine YENI GOVDEYLE FIZIKTE URETILMIS acilimlar (degerlendirme/
# video_fizik_opt.py, VIDEO_GOVDE_EK=0.09). Kinematik hedef fiziksel olarak takip edilemiyordu
# (taklit r/adim %11-19'da takili). Bagimsiz dogrulama: 171441 gurultulu 11/11 ayakta (saglam),
# 171405 1/11 (kirilgan ama fiziksel), 171522 kalkamadi -> cikarildi. Bu acilimlar YENI govdeyle:
# taklitte kok z duzeltmesi UYGULANMAZ (_mimic_ref_yeni).
YERDE_VIDEO_FILES = ("referanslar/video_171441_govde_fizik.npz",
                     "referanslar/video_171405_govde_fizik.npz")
YERDE_VIDEO_W     = (0.7, 0.3)    # saglam olana agirlik
YERDE_VIDEO_FRAC  = 0.7
if EYLEM_MODU == "artik":
    YERDE_VIDEO_FRAC = 1.0   # eski yatis pozlarinin (yuzustu dahil) referansi/tabani yok -> bu kosuda kapali
YERDE_SURE_S      = 10.0      # kullanici: robot insandan yavas olabilir (171522 ayakta 5.0 s; 6 -> 7 -> 10)
# (a) VIDEO YOLU (2026-10-14, kullanici onayi): 6M'de yerde 10/12 yatik, taklit r/adim 0.29/2.0
# (ilerlemedi); 4M'de robot kendi 'geriye takla -> dort ayak' yolunu buldu ama video yolunu
# hic denemedi. Yerde-video bolumlerinde taklit agirligi artirilir (diger modlarda MIMIC_W).
MIMIC_W_YERDE     = 5.0
MIMIC_OTURMA_FRAC = 0.30   # oturma eslesmesi: k150 reflerin ilk %30'u
MIMIC_MATCH_MAX = 1.0      # eslesme mesafesi esigi
# (a) FAZ BEKLETME (2026-10-13): ajanin oturmasi 'uzun oturma' (dizler duz, bacaklar
# yana acik) — video oturmasi 'diz cekili'; eslesme mesafesi medyan 10 (0/60 eslesme).
# Artik esik ustunde taklit KAPANMAZ: hedef en yakin k150 ref'in ILK karesi (diz cekili
# oturma) olur ve faz, mesafe esik altina inene dek BEKLER -> once 'bacaklari topla'
# odullenir, sonra kalkis takip edilir. (yayp olcumu: toplama PD ile 15/15, 1.2 s.)
# (b) OTURMA KARISIMI: oturma bolumlerinin OTURMA_VIDEO_FRAC'i k150 acilimlarinin ilk
# OTURMA_VIDEO_FIRST kismindan (diz cekili) baslar -> toplanmis->ayakta kapali-dongu
# kalkis pratigi (yayp: acik dongu video kalkisi 0/15, yani kapali dongu sart).
OTURMA_VIDEO_FRAC  = 0.5
# TOPLAMA SEKILLENDIRMESI (2026-10-13, yayp buldu): bekleme fazinda exp(-2*eklem^2) DUZDU —
# uzun oturmadan %50 toplamaya r_im 0.810 -> 0.823 (gradyan yok; log TOPLADI 0/345).
# Bekleme fazinda potansiyel tabanli terim: F = Phi(s') - Phi(s) (gamma=1, bkz. step), Phi = -TUCK_C*dist
# (dist = _pose_dist, diz cekili ilk kareye). Optimum politikayi degistirmez (Ng 1999).
TUCK_C = 0.5
OTURMA_VIDEO_FIRST = 0.15


def _yaw_quat(yaw):
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


_HAVUZ = {"mtime": None, "qpos": None, "qvel": None}
def _load_havuz():
    """Oturma havuzu — dosya degistikce (egitim scripti yeniler) yeniden yuklenir."""
    try:
        mt = os.path.getmtime(HAVUZ_FILE)
    except OSError:
        return None, None
    if mt != _HAVUZ["mtime"]:
        try:
            d = np.load(HAVUZ_FILE)
            _HAVUZ.update(mtime=mt, qpos=d["qpos"].astype(np.float64),
                          qvel=d["qvel"].astype(np.float64))
        except Exception:          # yazim aninda okunduysa: eski havuzla devam
            pass
    return _HAVUZ["qpos"], _HAVUZ["qvel"]


_YVIDEO = []
def _load_yerde_video():
    """[(qpos[T,19], dt)] — kinematik v2 tam zincir yerden kalkis videolari."""
    if not _YVIDEO:
        for f in YERDE_VIDEO_FILES:
            dd = np.load(f)
            _YVIDEO.append((dd["qpos"].astype(np.float64), float(dd["dt"]),
                            dd["qvel"].astype(np.float64) if "qvel" in dd.files else None))
    return _YVIDEO


# ── SOL/SAG AYNA SIMETRISI (2026-10-14, kullanici onayi) ──────────────────────
# Olculdu (eklem 4M): politika sirtustunden otururken SOL uylugu gogse cekip (kalca -2.62,
# sinirda) SAG bacagi aktif olarak DUZ tutuyor (diz -0.03, diz komutu +0.37 = acma yonu);
# hedef iki diz -2.49. Fizik kontrolu (degerlendirme/diz_cekme_fizik.py): robotun kendi
# oturusundan iki diz -2.48'e cekilebiliyor, govde dik kaliyor, govde-ici temas yok ->
# engel fiziksel degil, tek-tarafli aliskanlik. Ag sol/sag simetrisini zorunlu kilmadigi icin
# bu yerel optimum kirilmiyor. Cozum: simetri kaybi (Yu ve ark. 2018, MSL):
#   L_sym = SYM_W * mean( (mu(s) - M_a(mu(M_s(s))))^2 )
# M_s / M_a = gozlem / eylem ayna haritasi (izin + isaret). Ayna duzlemi: govde x-z (y -> -y).
# Haritalar degerlendirme/simetri_dogrula.py ile FIZIKSEL aynalanmis durumlarda dogrulandi.
SYM_W = 4.0

_AYNA_MODEL = {}
def _ayna_model():
    if "m" not in _AYNA_MODEL:
        _AYNA_MODEL["m"] = mujoco.MjModel.from_xml_path(XML)
    return _AYNA_MODEL["m"]


def _eklem_ayna(ad):
    """Eklem/aktüatör adı (p1_ öneki yok) → (ayna adı, işaret). x/z eksenli (yana açma, yaw) eklemler aynada işaret değiştirir."""
    es = ad.replace("left", "#").replace("right", "left").replace("#", "right")
    # govde2 (2026-10-21): omuz yana açma (x) ve bel (z) de işaret değiştirir; bel_z kendi aynası
    return es, (-1 if any(k in ad for k in ("hip_x", "hip_z", "shoulder_x", "bel_z")) else 1)


def ayna_haritalari():
    """(obs_perm, obs_sign, act_perm, act_sign): ayna(x)[i] = x[perm[i]] * sign[i]. 2026-10-18: eklem ve aktüatör
    sıraları XML'den İSİMLE kurulur (kalça yaw'lı robotta da geçerli); eski robotta eski sabit haritayla birebir aynı (K15)."""
    m = _ayna_model()
    J = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)[3:] for j in range(m.njnt)
         if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or "").startswith("p1_") and m.jnt_type[j] != 0]
    J = [J[i] for i in np.argsort([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "p1_" + a)] for a in J])]
    A = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)[3:] for i in range(m.nu)
         if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i).startswith("p1_")]
    nj = len(J)
    jp = [J.index(_eklem_ayna(a)[0]) for a in J]; js = [_eklem_ayna(a)[1] for a in J]
    perm, sign = [], []
    def ekle(idx, sg): perm.extend(idx); sign.extend(sg)
    V, W = (1, -1, 1), (-1, 1, -1)
    R = 10 + 2 * nj                                       # eski düzende 34 (eklem bloklarından sonraki kısım)
    ekle([0], [1]); ekle([1, 2, 3], V)
    ekle([4 + j for j in jp], js)
    ekle([4 + nj, 5 + nj, 6 + nj], V); ekle([7 + nj, 8 + nj, 9 + nj], W)
    ekle([10 + nj + j for j in jp], js)
    ekle([R, R + 1, R + 2], [1, 1, 1])
    ekle([R + 3, R + 4, R + 5], V); ekle([R + 6, R + 7, R + 8, R + 9], [1] * 4)
    ekle([R + 10, R + 11, R + 12], V)
    for a, b in ((13, 16), (19, 22), (25, 28), (31, 34), (37, 40), (43, 46), (49, 52), (55, 58)):
        ekle([R + b, R + b + 1, R + b + 2], V); ekle([R + a, R + a + 1, R + a + 2], V)
    ekle([R + 61, R + 62, R + 63, R + 64, R + 65], (1, -1, 1, 1, -1))
    ap = [A.index(_eklem_ayna(a)[0]) for a in A]; as_ = [_eklem_ayna(a)[1] for a in A]
    return (np.array(perm), np.array(sign, np.float32), np.array(ap), np.array(as_, np.float32))


def eski_gozlem_idx():
    """Eski düzen GetupEnv gözlemi (100) → bu robotun gözlemindeki indeksler (eklem blokları isimle eşlenir)."""
    m = _ayna_model()
    J = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)[3:] for j in range(m.njnt)
         if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or "").startswith("p1_") and m.jnt_type[j] != 0]
    J = [J[i] for i in np.argsort([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "p1_" + a)] for a in J])]
    nj = len(J); e = [J.index(a) for a in ESKI_EKLEMLER]
    return np.array([0, 1, 2, 3] + [4 + i for i in e] + [4 + nj + i for i in range(6)] + [10 + nj + i for i in e]
                    + [10 + 2 * nj + i for i in range(66)])


def _ayna_haritalari_eski_sabit():
    """(obs_perm, obs_sign, act_perm, act_sign): ayna(x)[i] = x[perm[i]] * sign[i]. ESKİ SABİT SÜRÜM (yalnız K15 karşılaştırması)."""
    # eklem sirasi (qpos[7:19]): 0 omuz_s 1 dirsek_s 2 omuz_l 3 dirsek_l 4 kalca_x_l 5 kalca_y_l
    #   6 diz_l 7 bilek_l 8 kalca_x_s 9 kalca_y_s 10 diz_s 11 bilek_s   (s = sag, l = sol)
    jp = [2, 3, 0, 1, 8, 9, 10, 11, 4, 5, 6, 7]
    js = [1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1]           # kalca_x (eksen x) isaret degistirir
    perm, sign = [], []
    def ekle(idx, sg): perm.extend(idx); sign.extend(sg)
    V, W = (1, -1, 1), (-1, 1, -1)                        # vektor / acisal hiz (psodovektor)
    ekle([0], [1])                                        # yukseklik
    ekle([1, 2, 3], V)                                    # projeksiyon yercekimi
    ekle([4 + j for j in jp], js)                         # eklem acilari
    ekle([16, 17, 18], V); ekle([19, 20, 21], W)          # kok dogrusal / acisal hiz
    ekle([22 + j for j in jp], js)                        # eklem hizlari
    ekle([34, 35, 36], [1, 1, 1])                         # hp, rakip hp, mesafe
    ekle([37, 38, 39], V); ekle([40, 41, 42, 43], [1] * 4)   # rakip goreli konum, z, kombolar, zaman
    ekle([44, 45, 46], V)                                 # rakip hizi
    for a, b in ((47, 50), (53, 56), (59, 62), (65, 68), (71, 74), (77, 80), (83, 86), (89, 92)):
        ekle([b, b + 1, b + 2], V); ekle([a, a + 1, a + 2], V)   # sol <-> sag uzuv konumlari
    ekle([95, 96, 97, 98, 99], (1, -1, 1, 1, -1))         # kutle merkezi - ayak, hiz
    # eylem (aktuator) sirasi: 0 omuz_s 1 dirsek_s 2 omuz_l 3 dirsek_l 4 kalca_x_l 5 kalca_y_l
    #   6 diz_l 7 kalca_x_s 8 kalca_y_s 9 diz_s 10 bilek_l 11 bilek_s
    ap = [2, 3, 0, 1, 7, 8, 9, 4, 5, 6, 11, 10]
    as_ = [1, 1, 1, 1, -1, 1, 1, -1, 1, 1, 1, 1]
    return (np.array(perm), np.array(sign, np.float32), np.array(ap), np.array(as_, np.float32))


_YHEDEF = {}
def _yerde_pd_hedef(i, qa):
    """Fizik acilimini URETEN PD hedef dizisi [T,12] (aktuator sirasi) + bilek kazanci kc.
    video_fizik_opt.py hedef_uret ile ayni: anahtar pozlar arasi dogrusal, sureler kayittan."""
    if i not in _YHEDEF:
        R = np.load(YERDE_VIDEO_FILES[i]); Q0 = R["qpos"][0].astype(np.float64)
        ES = [(7, 9), (8, 10), (12, 16), (13, 17), (14, 18)]     # omuz, dirsek, kalca_y, diz, bilek (qpos)
        H = []
        for k in range(len(R["anahtar"]) - 1):
            n_ = max(2, int(round(float(R["sure"][k]) / PHYS_DT)))
            for s_ in range(n_):
                t = s_ / n_; h = Q0.copy()
                for j, (a_, b_) in enumerate(ES):
                    h[a_] = h[b_] = (1 - t) * R["anahtar"][k, j] + t * R["anahtar"][k + 1, j]
                H.append(h)
        qa = np.asarray(qa); H = np.array(H); Hq = np.zeros((len(H), len(qa)))
        Hq[:, qa >= 0] = H[:, qa[qa >= 0]]                      # qa: aktüatör → eski düzen qpos indeksi (yeni eklem -1 → hedef 0)
        _YHEDEF[i] = (Hq, float(R["kc"]))
    return _YHEDEF[i]


_FAZ_OZ = {}
def _faz_ozellik(env, Q):
    """Faz aramasi icin ref kare ozellikleri (eklem, kok z, egim) — ref dizisi basina bir kez hesaplanir.
    Not: eski-govde ref'lerinde kok z duzeltmesi env._mimic_ref_yeni'ye bagli; anahtar buna gore."""
    anahtar = (id(Q), bool(getattr(env, "_mimic_ref_yeni", False)))
    if anahtar not in _FAZ_OZ:
        f = [env._pose_feats(qq, ref=True) for qq in Q]
        _FAZ_OZ[anahtar] = (np.array([x[0] for x in f]), np.array([x[1] for x in f]), np.array([x[2] for x in f]))
    return _FAZ_OZ[anahtar]


_TOPLAMA = {}
def _load_toplama():
    if not _TOPLAMA:
        d = np.load(TOPLAMA_FILE)
        _TOPLAMA.update(qpos=d["qpos"].astype(np.float64), qvel=d["qvel"].astype(np.float64))
    return _TOPLAMA["qpos"], _TOPLAMA["qvel"]


_REFS = {}
def _load_refs(files=REF_FILES):
    """Referans kalkma yorungeleri (qpos[T,19], qvel[T,18]) — dosya listesi basina bir kez."""
    if files not in _REFS:
        _REFS[files] = []
        for f in files:
            d = np.load(f)
            _REFS[files].append((d["qpos"].astype(np.float64), d["qvel"].astype(np.float64)))
    return _REFS[files]


class GetupEnv(st.DualHumanoidFightEnv):
    """Sadece KALKMA ortami. Ayni XML/gozlem, kendi reset/odul/sonlandirma."""

    def __init__(self, env_id=0, push_min=PUSH_MIN, push_max=PUSH_MAX, xml=XML):
        super().__init__(xml_path=xml, env_id=env_id, difficulty=1.0)
        # dusuk-cozunurluk fizik (bkz. PHYS_DT yorumu) — XML'e dokunmadan
        self.model.opt.timestep = PHYS_DT
        self.model.opt.iterations = SOLVER_ITER
        # motorsuz bilekte (eski XML) pasif yayi guclendir; motorlu XML'e dokunma
        self.ankle_motor = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "p1_ankle_left") >= 0
        if not self.ankle_motor:
            for side in ("left", "right"):
                j = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"p1_ankle_{side}")
                self.model.jnt_stiffness[j] = ANKLE_STIFFNESS
        # GOZLEM EKI: kutle merkezi - ayak (yayp-bilgisayar onerisi; taramasinda
        # denge tam bu sinyalle saglandi). Ego cercevede 5 boyut, bkz. _com_features.
        m = self.model
        self._p1_bodies = [b for b in range(m.nbody) if (mujoco.mj_id2name(
            m, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith("p1")]
        self._p1_mass = m.body_mass[self._p1_bodies]
        self._prev_com = None
        self._eski_duzen_kur()
        n_obs = self.get_obs_both()[0].shape[0]
        self.observation_space = st.spaces.Box(low=-np.inf, high=np.inf,
                                               shape=(n_obs,), dtype=np.float32)
        self.max_timesteps = MAX_STEPS
        self.push_min = float(push_min)
        self.push_max = float(push_max)
        self.last_push = 0.0
        self.assist_force = 0.0        # N; egitim scripti ayarlar, varsayilan KUVVETSIZ
        self.assist_steps = 0          # bu episode'da kuvvetin aktif oldugu adim sayisi
        self._prev_phi = 0.0
        self._rsi_pick, self._rsi_k = None, 0
        self._mimic_Q, self._mimic_k = None, 0
        self.mimic_sum, self.mimic_on, self.mimic_dist = 0.0, False, float('nan')
        self._mimic_hold, self.mimic_released, self.oturma_src = False, False, None
        self._tuck_phi, self.tuck_sum = 0.0, 0.0
        self._yerde_video, self._mimic_step, self._mimic_w = None, MIMIC_REF_STEP, MIMIC_W

    # ── ESKİ DÜZEN görünümü (2026-10-18, kalça yaw planı) ─────────────
    def _eski_duzen_kur(self):
        """q19/v18: eski düzen (kök 7 + ESKI_EKLEMLER) mutlak adresleri; yeni eklemler ayrı. Eski robotta birim eşleme."""
        m = self.model; jid = lambda a: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "p1_" + a)
        p1s, vs = self.p1_qpos_start, self.p1_root_dof_adr
        self._q19 = np.r_[np.arange(p1s, p1s + 7), [m.jnt_qposadr[jid(a)] for a in ESKI_EKLEMLER]]
        self._v18 = np.r_[np.arange(vs, vs + 6), [m.jnt_dofadr[jid(a)] for a in ESKI_EKLEMLER]]
        tum_q = np.arange(p1s, p1s + self.per_player_qpos_dim); tum_v = np.arange(vs, vs + self.per_player_vel_dim)
        self._q_yeni = np.setdiff1d(tum_q, self._q19); self._v_yeni = np.setdiff1d(tum_v, self._v18)
        # aktüatör → eski düzen qpos indeksi (0..18; yeni eklem -1): eski referanslardan PD hedefi kurmak için
        q_rel = {int(self._q19[i]): i for i in range(19)}
        self._akt_q19 = np.array([q_rel.get(int(m.jnt_qposadr[m.actuator_trnid[i, 0]]), -1)
                                  for i in range(self.p1_actuator_count)])

    def q19(self):
        return self.data.qpos[self._q19].copy()

    def q19_yaz(self, q):
        self.data.qpos[self._q19] = q; self.data.qpos[self._q_yeni] = 0.0

    def v18(self):
        return self.data.qvel[self._v18].copy()

    def v18_yaz(self, v):
        self.data.qvel[self._v18] = v; self.data.qvel[self._v_yeni] = 0.0

    def eski_hedef(self, q19):
        """Eski düzen poz → PD hedefi (aktüatör sırası); eski düzende olmayan eklemlerin hedefi 0."""
        h = np.zeros(len(self._akt_q19)); k = self._akt_q19 >= 0; h[k] = np.asarray(q19)[self._akt_q19[k]]
        return h

    # ── kendi reset: DIK basla + rastgele itme (dogal dusus) ──────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p1s = self.p1_qpos_start
        d = self.p1_root_dof_adr
        # P2 uzakta, alakasiz (gozlem boyutu korunsun diye modelde duruyor)
        if self.p2_qpos_start is not None:
            self.data.qpos[self.p2_qpos_start:self.p2_qpos_start + 3] = [8.0, 0.0, 0.62 + GOVDE_EK]

        mode = (options or {}).get("start") or np.random.choice(
            list(START_PROBS), p=list(START_PROBS.values()))
        if mode == "oturma":
            hq, hv = _load_havuz()
            if hq is None or len(hq) == 0:
                mode = "yerde"                 # havuz henuz yok -> yerde
        self.start_mode = mode
        self.last_push = 0.0
        yaw = float(np.random.uniform(0, 2 * np.pi))

        if mode in ("oturma", "toplama"):
            # oturma: ajanin kendi oturmasi / diz cekili video; toplama: uzun->diz cekili ara
            # durumlar. Hepsine kucuk gurultu + rastgele yon.
            if mode == "toplama":
                tq, tv = _load_toplama()
                i = np.random.randint(len(tq))
                q = tq[i].copy(); v = tv[i].copy()
                self.oturma_src = "toplama"
            elif np.random.rand() < OTURMA_VIDEO_FRAC:
                Qv, Vv = _load_refs(OTURMA_REF_FILES)[np.random.randint(len(OTURMA_REF_FILES))]
                i = np.random.randint(max(1, int(len(Qv) * OTURMA_VIDEO_FIRST)))
                q = Qv[i].copy(); v = Vv[i].copy()
                self.oturma_src = "video"
            else:
                i = np.random.randint(len(hq))
                q = hq[i].copy(); v = hv[i].copy()
                self.oturma_src = "havuz"
            qz = _yaw_quat(yaw); qn = np.zeros(4)
            mujoco.mju_mulQuat(qn, qz, q[3:7]); q[3:7] = qn / np.linalg.norm(qn)
            c, s = np.cos(yaw), np.sin(yaw)
            v[0], v[1] = c * v[0] - s * v[1], s * v[0] + c * v[1]
            q[0:2] = 0.0
            q[7:] += np.random.normal(0, HAVUZ_QPOS_NOISE, size=len(q) - 7)
            v += np.random.normal(0, HAVUZ_QVEL_NOISE, size=len(v))
            self.q19_yaz(q)
            if d >= 0:
                self.v18_yaz(v)
        elif mode == "itme":
            # P1 dik, eklemler notr + rastgele itme (dogal dusus)
            self.data.qpos[p1s:p1s + 3] = [0.0, 0.0, 0.62 + GOVDE_EK]
            self.data.qpos[p1s + 3:p1s + 7] = [1.0, 0.0, 0.0, 0.0]
            self.data.qpos[p1s + 7:p1s + self.per_player_qpos_dim] = 0.0
            if d >= 0:
                self.data.qvel[d:d + 6] = 0.0
                push = float(np.random.uniform(self.push_min, self.push_max))
                ang = float(np.random.uniform(0, 2 * np.pi))
                self.data.qvel[d + 0] = push * np.cos(ang)          # vx
                self.data.qvel[d + 1] = push * np.sin(ang)          # vy
                self.data.qvel[d + 3] = np.random.uniform(-1, 1) * push * 0.4
                self.data.qvel[d + 4] = np.random.uniform(-1, 1) * push * 0.4
                self.last_push = push
        else:
            # referans yorungeden durum: yerde = ilk kare, rsi = rastgele ara kare
            files = REF_FILES if mode in ("yerde", "zincir") else RSI_FILES
            self.start_ref = np.random.randint(len(files))
            qpos_ref, qvel_ref = _load_refs(files)[self.start_ref]
            self._yerde_video = None
            if mode == "zincir" or (mode == "yerde" and np.random.rand() < YERDE_VIDEO_FRAC):
                vi = int(np.random.choice(len(YERDE_VIDEO_FILES), p=YERDE_VIDEO_W))
                Qy, ydt, Vy = _load_yerde_video()[vi]
                self._yerde_video = (Qy, ydt)
                self.start_ref = 10 + vi                 # 10+ = video basl. (log icin)
                qpos_ref = Qy
                qvel_ref = Vy if (mode == "zincir" and Vy is not None) else np.zeros((len(Qy), 18))
            self._rsi_pick = qpos_ref
            T = len(qpos_ref)
            k = 0 if mode == "yerde" else np.random.randint(0, int(T * RSI_MAX_FRAC))
            self._zincir_k = k
            self._rsi_k = k
            q = qpos_ref[k].copy(); v = qvel_ref[k].copy()
            # rastgele yon (yaw): kok quat'i ve DUNYA-cercevesindeki dogrusal hizi dondur
            # (serbest eklemin acisal hizi govde cercevesinde -> degismez)
            qz = _yaw_quat(yaw); qn = np.zeros(4)
            mujoco.mju_mulQuat(qn, qz, q[3:7]); q[3:7] = qn / np.linalg.norm(qn)
            c, s = np.cos(yaw), np.sin(yaw)
            v[0], v[1] = c * v[0] - s * v[1], s * v[0] + c * v[1]
            q[0:2] = 0.0
            q[7:] += np.random.normal(0, JOINT_NOISE, size=len(q) - 7)
            self.q19_yaz(q)
            if d >= 0:
                self.v18_yaz(v if mode in ("rsi", "zincir") else np.zeros(18))

        mujoco.mj_forward(self.model, self.data)
        if mode != "itme":
            # referans kinematik -> bazi karelerde zemine ~16 cm gomulme olculdu
            # (ani temas kuvveti = firlama). Gomulmeyi kok-z'yi kaldirarak gider.
            for _ in range(3):
                pen = self._p1_min_contact_dist()
                if pen > -0.005:
                    break
                self.data.qpos[p1s + 2] += -pen - 0.002
                mujoco.mj_forward(self.model, self.data)
        if mode == "yerde":
            # sifir aksiyonla otur: penetrasyon/havada kalma cozulsun (fiziksel durum)
            self.data.ctrl[:] = 0.0
            for _ in range(SETTLE_STEPS):
                mujoco.mj_step(self.model, self.data)
        self.timestep = 0
        self.max_timesteps = (int(round(YERDE_SURE_S / (PHYS_DT * FRAME_SKIP)))
                              if mode in ("yerde", "zincir") else MAX_STEPS)
        self._prev_com = None          # yeni episode: COM hizi 0'dan
        self.assist_steps = 0
        self.data.xfrc_applied[self.p1_torso_id] = 0.0
        self._prev_phi = self._phi()
        self._mimic_setup(mode)
        obs_p1, _ = self.get_obs_both()
        return obs_p1, {}

    @staticmethod
    def _tilt(quat):
        """ISARETLI one egim (rad): + = one (yuzustu yonu), - = geriye. Yaw'dan bagimsiz.
        Eski surum isaretsizdi (acos R22): one 20 derece ile geriye 20 derece AYNI odulu
        aliyordu — kullanici gozlemi 'agirligini one vermiyor, arkaya dusuyor' tam bu ayrimi
        gerektiriyor. Saf pitch e icin: govde x(on)=(cos e,0,-sin e), z(yukari)=(sin e,0,cos e)
        -> atan2(-on_z, yukari_z) = e. Robot +x'e bakiyor (kalca bukulunce diz +x'e gidiyor)."""
        R = np.zeros(9); mujoco.mju_quat2Mat(R, np.asarray(quat, np.float64))
        return float(np.arctan2(-R[6], R[8]))

    def _pose_feats(self, q19, ref=False):
        t = self._tilt(q19[3:7]); z = float(q19[2])
        if ref and not getattr(self, "_mimic_ref_yeni", False):   # ESKI govde ref'i: kok GOVDE_EK*cos(egim) yukari
            z += GOVDE_EK * max(np.cos(t), 0.0)
        return q19[7:19][MIMIC_JOINTS], z, t

    def _mimic_reward(self, ref19):
        p1s = self.p1_qpos_start
        j, z, t = self._pose_feats(self.q19())
        jr, zr, tr = self._pose_feats(ref19, ref=True)
        # EKLEM TERIMI (2026-10-14): eskiden exp(-2*SUM(fark^2)) — bacaklar duz otururken her diz
        # ~1.9 rad uzakta -> exp(-2*7) ~ 1e-6: terim DUZ, ajan dizi cekmeye hic sinyal almiyordu
        # (zincir 4M: diz bukumu -0.51, hedef -2.40). Artik eklem BASINA genis cekirdek ortalamasi:
        # her eklem kendi kismi odulunu alir (diz 1.9 rad uzakta 0.16, 1.0 rad'da 0.61).
        return self._mimic_w * (0.5 * float(np.mean(np.exp(-MIMIC_EKLEM_K * (j - jr) ** 2)))
                          + 0.3 * np.exp(-40.0 * (z - zr) ** 2)
                          + 0.2 * np.exp(-10.0 * (t - tr) ** 2))

    def _pose_dist(self, q19, ref19):
        j, z, t = self._pose_feats(q19)
        jr, zr, tr = self._pose_feats(ref19, ref=True)
        return float(np.sum((j - jr) ** 2)) + 2.0 * (t - tr) ** 2 + 5.0 * (z - zr) ** 2

    def _mimic_setup(self, mode):
        """Bolum basinda taklit referansi + faz: rsi -> secilen ref/kare; oturma ->
        k150 reflerin ilk MIMIC_OTURMA_FRAC'inde en yakin kare (esik ustu: kapali)."""
        self._mimic_Q, self._mimic_k, self._mimic_hold = None, 0, False
        self._mimic_step = MIMIC_REF_STEP          # ref kare / kontrol adimi (kesirli olabilir)
        self._mimic_ref_yeni = False               # True: ref YENI govdeyle uretildi (kok z duzeltmesi yok)
        self._faz_takip = False                    # True: hedef saat yerine robotun fazina bagli (FAZ_*)
        self._taban_H, self._taban_e = None, None   # ARTIK: yerde/zincir PD hedef dizisi, bilek turev bellegi
        self._mimic_w = MIMIC_W                    # yerde-video: MIMIC_W_YERDE
        self.mimic_sum, self.mimic_on, self.mimic_dist = 0.0, False, float('nan')
        self.mimic_released = False
        self._tuck_phi, self.tuck_sum = 0.0, 0.0
        if mode in ("yerde", "zincir") and getattr(self, "_yerde_video", None) is not None:
            Qy, ydt = self._yerde_video
            self._mimic_Q, self._mimic_k = Qy, float(self._zincir_k if mode == "zincir" else 0)
            self._mimic_ref_yeni = True
            self._faz_takip = True
            self._pd_hazirla()
            self._taban_H, self._taban_kc = _yerde_pd_hedef(int(self.start_ref) - 10, self._akt_q19)
            self._faz_F = _faz_ozellik(self, Qy)              # faz aramasi icin ref ozellikleri (onbellekli)
            self._mimic_step = (PHYS_DT * FRAME_SKIP) / ydt   # 0.016/0.003 ~ 5.33
            self._mimic_w = MIMIC_W_YERDE
        elif mode == "rsi":
            self._mimic_Q, self._mimic_k = self._rsi_pick, int(self._rsi_k)
        elif mode in ("oturma", "toplama"):
            p1s = self.p1_qpos_start
            q19 = self.q19()
            best, first = (np.inf, None, 0), (np.inf, None)
            for Q, _V in _load_refs(OTURMA_REF_FILES):         # oturmadan kalkis refleri
                for k in range(0, int(len(Q) * MIMIC_OTURMA_FRAC), MIMIC_REF_STEP):
                    dist = self._pose_dist(q19, Q[k])
                    if dist < best[0]:
                        best = (dist, Q, k)
                    if k == 0 and dist < first[0]:
                        first = (dist, Q)
            self.mimic_dist = best[0]
            # 2026-10-17 (kullanici onayi): ARTIK eylemde SINIRSIZ 'faz bekletme' tabani DONDURUYORDU (diz cekili
            # oturus pozu; kalkis icin gereken degisim > ARTIK_PAY) -> robotun kendi oturus havuzundan
            # baslangiclar 0/19 kalkti. Denenenler: faz takibi (kinematik ref takip edilemedi, faz durdu: toplama
            # 31->11/36), beklemesiz zamanla ilerleme (oturma 36/36 ama toplama 11/36 — toplamada bekleme ise
            # yariyor). Cozum: bekleme EN FAZLA TUT_MAX_S, sonra taban mutlaka ilerler (step()).
            self._tut_sayac = 0
            if best[0] <= MIMIC_MATCH_MAX:
                self._mimic_Q, self._mimic_k = best[1], best[2]
            else:                                   # (a) faz bekletme: once topla
                self._mimic_Q, self._mimic_k, self._mimic_hold = first[1], 0, True
                self._tuck_phi = -TUCK_C * first[0]
        self.mimic_on = self._mimic_Q is not None

    def _phi(self):
        """Potansiyel: bas yuksekligi orani [0,1] (ayakta = 1)."""
        return min(float(self._get_player_position('p1')[2]) / STAND_HEAD_Z, 1.0)

    def _com_features(self):
        """[com-ayak_orta x,y (ego), com-ayak z, com hizi x,y (ego)] — 5 boyut."""
        d = self.data
        com = (d.xipos[self._p1_bodies] * self._p1_mass[:, None]).sum(0) / self._p1_mass.sum()
        feet = 0.5 * (d.xpos[self.key_body_ids['p1_foot_left']] +
                      d.xpos[self.key_body_ids['p1_foot_right']])
        vel = (np.zeros(3) if self._prev_com is None
               else (com - self._prev_com) / (PHYS_DT * FRAME_SKIP))
        self._prev_com = com.copy()
        xm = d.xmat[self.p1_torso_id]
        yaw = np.arctan2(xm[3], xm[0]); c, s_ = np.cos(yaw), np.sin(yaw)
        e = com - feet
        return np.array([c * e[0] + s_ * e[1], -s_ * e[0] + c * e[1], e[2],
                         c * vel[0] + s_ * vel[1], -s_ * vel[0] + c * vel[1]],
                        dtype=np.float32)

    def get_obs_both(self):
        o1, o2 = super().get_obs_both()
        return np.concatenate([o1, self._com_features()]).astype(np.float32), o2

    def _p1_min_contact_dist(self):
        """P1 geom'lari ile ZEMIN (world govdesi) arasindaki en negatif mesafe
        (gomulme, m). Oz-temaslar sayilmaz (kok-z kaldirmak onlari cozmez)."""
        m = self.model
        if not hasattr(self, "_p1_geoms"):
            self._p1_geoms = {g for g in range(m.ngeom) if (mujoco.mj_id2name(
                m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or "").startswith("p1")}
        P, wb = self._p1_geoms, m.geom_bodyid
        dists = [c.dist for c in self.data.contact[:self.data.ncon]
                 if (c.geom1 in P and wb[c.geom2] == 0) or (c.geom2 in P and wb[c.geom1] == 0)]
        return min(dists) if dists else 0.0

    # ── kendi odulu ──────────────────────────────────────────────────
    # KOK-FIX (2026-08, olcumle bulundu): ESKI odul YATARKEN +1.1..+1.4 veriyordu!
    # Sebep: yere yatinca AYAKLAR DA yere degiyor (ayak_z~0.07<0.18) -> planted=True
    # -> duz '+1.0 ayaklar yerde' bonusu + bas-yuksekliginden kismi puan. 800 adim
    # x 1.3 = ~1040 = gozlenen 'odul=1000-1400'. Yani ajan YATARAK garantili gelir
    # elde ediyordu; kalkmak zor+riskli, yatmak bedava -> 330 episode sifir ilerleme.
    # YENI: tek surekli 'ilerleme' terimi. YATMAK NEGATIF, ayakta guclu pozitif,
    # havada/firlama kesin negatif. Arada suruklu gradyan (kalkarken artan odul).
    def getup_reward(self):
        z        = float(self._get_player_position('p1')[2])
        torso_up = float(self.data.xmat[self.p1_torso_id][8])
        vz       = float(self.data.qvel[self.p1_root_dof_adr + 2]) if self.p1_root_dof_adr >= 0 else 0.0
        flz = float(self.data.xpos[self.key_body_ids['p1_foot_left']][2])
        frz = float(self.data.xpos[self.key_body_ids['p1_foot_right']][2])
        planted = min(flz, frz) < FOOT_PLANTED_Z

        # ilerleme: 0 = yerde yatik, 1 = tam dik ayakta
        # 2026-10 (olculdu, k120s ep300): eski prog = 0.5*dik + 0.5*yukseklik,
        # OTURMAYA ayakta odulunun %62'sini, COMELMEYE %78'ini veriyordu (ikisi
        # de cok daha az riskli) -> itme/rsi'de 8/10 comelip kaliyor, yerde'den
        # oturup kaliyordu. Yukseklik artik KARESEL ve agirlikli + gercek ayakta
        # durusa bonus: siralama korunur (yatik < oturma < comelme < ayakta,
        # gradyan surekli) ama ayakta acik ara en iyi.
        h = min(z / STAND_HEAD_Z, 1.0)
        prog = 0.3 * max(torso_up, 0.0) + 0.7 * h * h
        standing = z > STAND_Z and torso_up > STAND_TORSO

        if abs(vz) < CALM_VZ and planted:
            # yatik -1.0 | oturma +1.8 | comelme +2.4 | dik ayakta +5.8 (+2 bonus = 7.8)
            # (havadaki dal her zaman <= -1.3 -> yatik halden de kotu)
            r = 7.0 * prog - 1.2 + (STAND_BONUS if standing else 0.0)
        else:
            # ayaklar havada VEYA hizli dikey hareket (firlama) -> kesin negatif.
            # Taban yatmaktan (-1.0) KOTU olmali: eski -0.5 'sirtustu yatip
            # ayaklari havaya kaldir' yerel optimumunu yaratiyordu.
            r = -1.3 - 0.3 * abs(vz)

        act = self.data.ctrl[:self.p1_actuator_count]
        r -= 0.02 * float(np.sum(np.square(act)))            # kontrol cezasi
        return r

    # ── kendi step: dovus mantigi YOK, sadece timeout ─────────────────
    # FRAME_SKIP (2026-08): eskiden her 0.003 s'de bir aksiyon = saniyede 333
    # KARAR (absurd yuksek; standart Gym Humanoid-v4 frame_skip=5 -> 66 Hz).
    # Olcum: politika cagrisi toplam maliyetin %65'i (fizik %35). frame_skip=5
    # -> politika cagrisi 5 kat azalir (~2.1x hiz) VE kredi atamasi cok kolaylasir
    # (800 karar yerine 160 karar) — uzun-ufuklu getup icin asil fayda bu.
    def _pd_hazirla(self):
        if getattr(self, "_pd_qa", None) is not None:
            return
        m, n = self.model, self.p1_actuator_count
        aj = [m.actuator_trnid[i, 0] for i in range(n)]
        self._pd_qa = np.array([m.jnt_qposadr[j] for j in aj]); self._pd_va = np.array([m.jnt_dofadr[j] for j in aj])
        self._pd_g = m.actuator_gear[:n, 0].copy()
        self._pd_lo, self._pd_hi = m.jnt_range[aj, 0].copy(), m.jnt_range[aj, 1].copy()
        self._pd_orta, self._pd_yari = 0.5 * (self._pd_lo + self._pd_hi), 0.5 * (self._pd_hi - self._pd_lo)
        ad_ = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in aj]
        self._pd_bilek = [ad_.index("p1_ankle_left"), ad_.index("p1_ankle_right")]
        self._pd_kp, self._pd_kd = pd_kazanc(ad_)
        self._pd_ayak = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"p1_foot_{t}") for t in ("left", "right")]

    def taban_hedef(self):
        """ARTIK modunda eylem 0'in hedefi (aktuator sirasi): referans hareketin o anki PD hedefi."""
        self._pd_hazirla()
        H = getattr(self, "_taban_H", None)
        if H is not None and self._mimic_Q is not None:
            h = H[min(int(self._mimic_k) + TABAN_ONDE, len(H) - 1)].copy(); kc = self._taban_kc
        elif self._mimic_Q is not None:
            # faz takibinde taban fazin ONUNDEN (onde degilse taban = robotun pozu -> robot yerinde kalir,
            # faz hic ilerlemez: ayni donma baska yoldan geri gelir)
            k = (min(int(self._mimic_k) + FAZ_ONDE, len(self._mimic_Q) - 1) if self._faz_takip
                 else 0 if self._mimic_hold else min(int(self._mimic_k), len(self._mimic_Q) - 1))
            h = self.eski_hedef(self._mimic_Q[k]).astype(np.float64); kc = TABAN_KC
        else:
            h = np.zeros(self.p1_actuator_count); kc = TABAN_KC   # ayakta durus (itme baslangic pozu)
        # bilek dengesi (aciliimi ureten denetleyici): govde dikken kutle merkezini ayaklarin ustunde tut.
        # Olculdu: dengesiz 'eklemler 0' ayakta tabani 1 s sonra geriye devriliyor (com-ayak -0.03 -> -0.31 m).
        if TABAN_BILEK and float(self.data.xmat[self.p1_torso_id][8]) > 0.6:
            e = self._com_x_hata(); ev = (e - self._taban_e) / PHYS_DT if self._taban_e is not None else 0.0
            h[self._pd_bilek] = np.clip(h[self._pd_bilek] + kc * e + 0.5 * ev, -0.6, 0.6)
        return h

    def _com_x_hata(self):
        """Kutle merkezi x - ayaklarin ortasi x (bilek dengesi; video_fizik_opt.com_hata ile ayni)."""
        d = self.data; k = self._p1_mass
        return float((d.xipos[self._p1_bodies][:, 0] * k).sum() / k.sum() - d.geom_xpos[self._pd_ayak][:, 0].mean())

    def eylem_hedef(self, a):
        """PD eylemi [-1,1] -> hedef eklem acisi (aktuator sirasi)."""
        self._pd_hazirla()
        a = np.asarray(a, np.float64)
        if EYLEM_MODU == "artik":
            return np.clip(self.taban_hedef() + a * ARTIK_PAY, self._pd_lo, self._pd_hi)
        return np.clip(self._pd_orta + a * PD_ASIM * self._pd_yari, self._pd_lo, self._pd_hi)

    def hedef_eylem(self, hedef):
        """Hedef eklem acisi -> PD eylemi (eylem_hedef'in tersi; teshis/test icin)."""
        self._pd_hazirla()
        if EYLEM_MODU == "artik":
            return np.clip((np.asarray(hedef, np.float64) - self.taban_hedef()) / ARTIK_PAY, -1, 1)
        return np.clip((np.asarray(hedef, np.float64) - self._pd_orta) / (PD_ASIM * self._pd_yari), -1, 1)

    def step(self, action_p1, hedef=None):
        """hedef (2026-10-17, damitma): verilirse PD hedef acisi (aktuator sirasi) DOGRUDAN uygulanir, eylem->hedef
        eslemesi atlanir (tek ag mutlak hedef uretir); taban/bilek defteri (_taban_e) yine guncellenir."""
        self.timestep += 1
        a = np.clip(np.asarray(action_p1, np.float64), -1, 1)
        if PD_KONTROL:
            self._pd_hazirla()
            hedef = self.eylem_hedef(a) if hedef is None else np.clip(np.asarray(hedef, np.float64), self._pd_lo, self._pd_hi)
            if EYLEM_MODU == "artik":
                self._taban_e = self._com_x_hata()
        else:
            self.data.ctrl[:self.p1_actuator_count] = a
        self.data.ctrl[self.p1_actuator_count:] = 0.0        # P2 hareketsiz
        # YARDIM KUVVETI: yalniz govde dik + bas alcakken (oturma/comelme), yukari
        f = 0.0
        if self.assist_force > 0.0:
            tu = float(self.data.xmat[self.p1_torso_id][8])
            z = float(self._get_player_position('p1')[2])
            if tu > ASSIST_TORSO and z < ASSIST_HEAD_Z:
                k = min(1.0, (ASSIST_HEAD_Z - z) / (ASSIST_HEAD_Z - ASSIST_HEAD_FULL))
                f = self.assist_force * k
                self.assist_steps += 1
        self.data.xfrc_applied[self.p1_torso_id] = [0.0, 0.0, f, 0.0, 0.0, 0.0]
        for _ in range(FRAME_SKIP):                          # ayni aksiyonu tut
            if PD_KONTROL:                                   # PD her fizik adiminda yeniden
                d_ = self.data
                d_.ctrl[:self.p1_actuator_count] = np.clip(
                    (self._pd_kp * (hedef - d_.qpos[self._pd_qa]) - self._pd_kd * d_.qvel[self._pd_va]) / self._pd_g, -1, 1)
            mujoco.mj_step(self.model, self.data)

        reward = self.getup_reward()
        phi = self._phi()
        reward += POTENTIAL_W * (POTENTIAL_GAMMA * phi - self._prev_phi)
        self._prev_phi = phi            # bas potansiyeli HEMEN kaydedilir (asagidaki toplama
                                        # potansiyeli ayri degiskende — 2026-10-13 isim cakismasi
                                        # bas potansiyelini -5'e eziyor, bolumde +7000..+15000 sahte odul)
        if self._mimic_Q is not None:
            if self._mimic_hold:
                p1s = self.p1_qpos_start
                dist = self._pose_dist(self.q19(), self._mimic_Q[0])
                tphi = -TUCK_C * dist               # toplama potansiyeli: diz cekili poza yakinlik
                # gamma=1 (2026-10-13 duzeltme): gamma*Phi'-Phi, Phi~-5 iken her adim
                # (gamma-1)*Phi ~ +0.05 'yerinde dur' odulu veriyordu (bolumde ~+14,
                # toplamadan); toplayinca bu akis kesildigi icin TOPLAMAMAYI odullendiriyordu.
                # gamma=1: toplam = TUCK_C*(d0 - d_son) tam teleskop, yerinde durmak 0.
                f_tuck = tphi - self._tuck_phi
                self._tuck_phi = tphi
                reward += f_tuck; self.tuck_sum += f_tuck
                self._tut_sayac = getattr(self, "_tut_sayac", 0) + 1
                if dist <= MIMIC_MATCH_MAX or (EYLEM_MODU == "artik"
                                               and self._tut_sayac * PHYS_DT * FRAME_SKIP >= TUT_MAX_S):
                    self._mimic_hold = False; self.mimic_released = True   # topladi -> faz baslar
            elif self._faz_takip:
                p1s = self.p1_qpos_start; q = self.q19()
                kk = int(self._mimic_k); son = min(len(self._mimic_Q) - 1, kk + FAZ_PENCERE)
                FJ, FZ, FT = self._faz_F; j, z, t = self._pose_feats(q)
                dd = (((FJ[kk:son + 1] - j) ** 2).sum(1) + 2.0 * (FT[kk:son + 1] - t) ** 2
                      + 5.0 * (FZ[kk:son + 1] - z) ** 2)            # = _pose_dist, vektorel
                self._mimic_k = kk + min(int(np.argmin(dd)), FAZ_MAX_ILERI)
            else:
                self._mimic_k = min(self._mimic_k + self._mimic_step, len(self._mimic_Q) - 1)
            hedef_k = (min(int(self._mimic_k) + FAZ_ONDE, len(self._mimic_Q) - 1) if self._faz_takip
                       else int(self._mimic_k))
            r_im = self._mimic_reward(self._mimic_Q[hedef_k])
            reward += r_im
            self.mimic_sum += r_im
        terminated = False                                    # dusme BITIRMEZ
        truncated  = self.timestep >= self.max_timesteps
        obs_p1, _ = self.get_obs_both()
        return obs_p1, reward, terminated, truncated, {}


def rollout(envs, agent):
    """Her ortamda bir episode topla. -> per-env trajektoriler + istatistik."""
    n = len(envs)
    S = [[] for _ in range(n)]; A = [[] for _ in range(n)]
    LP = [[] for _ in range(n)]; R = [[] for _ in range(n)]
    V = [[] for _ in range(n)]; D = [[] for _ in range(n)]
    last_obs = [None] * n
    stats = dict(stand=0, fell=0, recovered=0, launch=0, ep_ret=[])

    for i, env in enumerate(envs):
        obs, _ = env.reset()
        done = False
        zmin, zmax = 9.9, 0.0
        ep_r = 0.0
        while not done:
            act, lp = agent.get_action(obs, training=True)
            val = agent.get_value(obs)
            nobs, rew, term, trunc, _ = env.step(act)
            S[i].append(obs); A[i].append(act); LP[i].append(lp)
            R[i].append(rew); V[i].append(val); D[i].append(1 if term else 0)
            z = float(env._get_player_position('p1')[2])
            zmin = min(zmin, z); zmax = max(zmax, z)
            ep_r += rew
            obs = nobs
            done = term or trunc
        last_obs[i] = obs
        # episode sonu degerlendirme
        z = float(env._get_player_position('p1')[2])
        tu = float(env.data.xmat[env.p1_torso_id][8])
        ends_up = z > STAND_Z and tu > STAND_TORSO
        if ends_up: stats['stand'] += 1
        if zmin < 0.40 + GOVDE_EK:
            stats['fell'] += 1
            if ends_up: stats['recovered'] += 1
        if zmax > 1.3: stats['launch'] += 1
        stats['ep_ret'].append(ep_r)
    return S, A, LP, R, V, D, last_obs, stats


def main():
    # --sifirdan : hicbir taban yukleme (yuruyucunun ziplama-onyargisi tasinmasin)
    # --resume X : belirli checkpoint'ten devam
    scratch = "--sifirdan" in sys.argv
    resume = None
    for i, a in enumerate(sys.argv):
        if a == "--resume" and i + 1 < len(sys.argv):
            resume = sys.argv[i + 1]
    if resume is None and not scratch:
        cks = sorted(glob.glob(CKPT_PREFIX + "*.pth"), key=os.path.getmtime)
        resume = cks[-1] if cks else "EN_IYI_YURUYUCU_ep3200.pth"

    envs = [GetupEnv(env_id=i) for i in range(NUM_ENVS)]
    obs_dim = envs[0].observation_space.shape[0]
    act_dim = envs[0].p1_actuator_count
    agent = st.CUDAPPOAgent(obs_dim, act_dim, name="GetupSolo")
    warmup = WARMUP_UPD
    if scratch:
        print("SIFIRDAN egitim (taban YOK — yuruyucu onyargisi tasinmiyor)", flush=True)
        warmup = 0                       # taze critic zaten; warmup gereksiz
    elif os.path.exists(resume):
        agent.load(resume)
        print(f"Taban yuklendi: {resume}", flush=True)
    else:
        print(f"! Taban bulunamadi ({resume}) — sifirdan.", flush=True)

    print(f"AYRI GETUP EGITIMI basladi. envs={NUM_ENVS} steps={MAX_STEPS} "
          f"itme=uniform({PUSH_MIN:.1f},{PUSH_MAX:.1f}) RAMPA-YOK", flush=True)

    ep = 0
    while True:
        ep += 1
        S, A, LP, R, V, D, last_obs, stats = rollout(envs, agent)

        # GAE (truncation -> critic bootstrap)
        states = []; actions = []; logps = []; rets = []; advs = []; vals = []
        for i in range(NUM_ENVS):
            if not S[i]:
                continue
            nv = agent.get_value(last_obs[i])          # hep truncated (timeout)
            adv = agent.compute_gae(R[i], V[i], nv, D[i])
            ret = [a + v for a, v in zip(adv, V[i])]
            states.extend(S[i]); actions.extend(A[i]); logps.extend(LP[i])
            rets.extend(ret); advs.extend(adv); vals.extend(V[i])

        advs_arr = np.array(advs, dtype=np.float32)
        if len(advs_arr) > 1:
            advs_arr = (advs_arr - advs_arr.mean()) / (advs_arr.std() + 1e-8)
        train_actor = (warmup <= 0)
        agent.update(states, actions, logps, np.array(rets, np.float32), advs_arr,
                     old_values=vals, batch_size=1024, epochs=4,
                     train_actor=train_actor)
        if warmup > 0:
            warmup -= 1
            if warmup == 0:
                print(f"  [ep{ep}] critic kalibre — actor artik egitiliyor", flush=True)

        # RAMPA YOK — itme her episode uniform(PUSH_MIN, PUSH_MAX) (env.reset icinde)
        stand_rate = stats['stand'] / NUM_ENVS
        rec_rate = stats['recovered'] / stats['fell'] if stats['fell'] else 0.0

        if ep % 5 == 0 or ep == 1:
            print(f"ep{ep:5d} itme={PUSH_MIN:.1f}-{PUSH_MAX:.1f} "
                  f"ayakta={stand_rate*100:.0f}% "
                  f"dusen={stats['fell']}/{NUM_ENVS} kalkan={stats['recovered']}/"
                  f"{max(stats['fell'],1)} firlama={stats['launch']} "
                  f"odul={np.mean(stats['ep_ret']):.0f}", flush=True)
        if rec_rate >= 0.3 and stats['fell'] >= 3:
            print(f"KALKMA ep{ep}: dusup-kalkma={rec_rate*100:.0f}% "
                  f"({stats['recovered']}/{stats['fell']})", flush=True)

        if ep % SAVE_FREQ == 0:
            agent.save(f"{CKPT_PREFIX}{ep}.pth")


if __name__ == "__main__":
    main()
