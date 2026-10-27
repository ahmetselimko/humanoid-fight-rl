# -*- coding: utf-8 -*-
"""Taktik ağ eğitimi (PLAN Faz 3, 2026-10-22; kullanıcı seçimi B): kategorik PPO + self-play, önce kurallardan taklit (BC).

  set ROBOT=govde2 & venv\\Scripts\\python.exe taktik_train.py --parca <…_komsu3_fizik.xml> --dunya arena
        [--alt bulut_sonuc/alt_dovus2_govde2_eppoz6M.pth] [--workers 4] [--bc-mac 40] [--guncelleme 300] [--mac 2]

Akış: (1) BC — kural vs kural maçlarından (gözlem, kural komutu) toplanır, ağ çapraz entropiyle öğrenir;
(2) PPO — öğrenen tarafı rastgele (P1/P2), rakip %50 kural, %50 havuzdan geçmiş sürüm; GAE (γ .99, λ .95), kırpma .2;
(3) her --olcum-her güncellemede kurala karşı 16 maç (açgözlü): kazanma oranı, isabet, bölge değişimi; kayıt taktik_ep*.pth.
Kazanma: (verilen − alınan hasar) + 10·(rakibin devrilmesi − kendi devrilmesi) > 0. Başarı ölçütü (plan): kurala %55+.
"""
import os, sys, time, math, argparse, random
import numpy as np, torch, torch.nn as nn, torch.multiprocessing as mp

KOK = os.path.dirname(os.path.abspath(__file__))


class Ag(nn.Module):
    def __init__(self, obs=47, n=9, g=256):
        super().__init__()
        self.govde = nn.Sequential(nn.Linear(obs, g), nn.Tanh(), nn.Linear(g, g), nn.Tanh())
        self.pi, self.v = nn.Linear(g, n), nn.Linear(g, 1)

    def forward(self, x):
        h = self.govde(x)
        return self.pi(h), self.v(h).squeeze(-1)


def _sec(ag, o, acgozlu):
    with torch.no_grad():
        lg, v = ag(torch.as_tensor(o[None]))
        dist = torch.distributions.Categorical(logits=lg)
        a = int(lg.argmax()) if acgozlu else int(dist.sample())
        return a, float(dist.log_prob(torch.tensor([a]))), float(v)


def skor(ist, i):
    return (ist["hasar"][i] - ist["hasar"][1 - i]) + 10.0 * (ist["devrilme"][1 - i] - ist["devrilme"][i])


def isci(conn, wid, args):
    os.chdir(KOK); sys.path.insert(0, KOK)
    torch.set_num_threads(1); np.random.seed(1000 + wid); random.seed(1000 + wid)
    import selfplay_training as st
    st.DEVICE = torch.device("cpu"); st.USE_AMP = False
    import taktik_env as TK
    env = TK.TaktikEnv(args.parca, args.alt, args.dunya, env_id=200 + wid)
    if args.kalkis:
        env.kalkis_yukle(args.kalkis)
    if args.donme:
        env.donme_yukle(args.donme)
    if args.gez:
        env.gez_yukle(args.gez)
    ben, rakip_ag = Ag(TK.OBS, TK.N_KOMUT), Ag(TK.OBS, TK.N_KOMUT)
    while True:
        msg = conn.recv()
        if msg is None:
            break
        out = []
        if msg["tur"] == "bc":
            for _ in range(msg["n"]):
                o = env.reset(); bitti = False; X, Y = [], []
                while not bitti:
                    k = [TK.kural_komut(env, 0), TK.kural_komut(env, 1)]
                    X += [o[0], o[1]]; Y += k
                    *o, _r1, _r2, bitti, _ = env.adim(*k); o = o[:2]
                out.append((np.array(X, np.float32), np.array(Y, np.int64)))
        else:
            ben.load_state_dict(msg["w"])
            if msg.get("rakip_w") is not None:
                rakip_ag.load_state_dict(msg["rakip_w"])
            for _ in range(msg["n"]):
                i = random.randint(0, 1) if not msg.get("acgozlu") else 0
                o = env.reset(); bitti = False; T = dict(o=[], a=[], lp=[], v=[], r=[])
                if args.gez:                                  # kurala karşı yalnız öğrenen (taktik_olc ile aynı), self-play'de ikisi
                    env.gez_oyuncular = {i} if msg.get("rakip_w") is None else {0, 1}
                env.vurus_fren = {i} if msg.get("rakip_w") is None else {0, 1}   # VURUS_FREN_HIZ: taktik_olc ile aynı
                while not bitti:
                    a, lp, v = _sec(ben, o[i], msg.get("acgozlu", False))
                    b = TK.kural_komut(env, 1 - i) if msg.get("rakip_w") is None else _sec(rakip_ag, o[1 - i], False)[0]
                    k = (a, b) if i == 0 else (b, a)
                    o1, o2, r1, r2, bitti, ist = env.adim(*k)
                    T["o"].append(o[i]); T["a"].append(a); T["lp"].append(lp); T["v"].append(v); T["r"].append((r1, r2)[i])
                    o = (o1, o2)
                T = {k_: np.array(v_, np.float32 if k_ != "a" else np.int64) for k_, v_ in T.items()}
                T["bilgi"] = dict(skor=skor(ist, i), isabet=ist["isabet"][i], alinan=ist["isabet"][1 - i],
                                  bolge=ist["bolge_degisimi"][i] + ist["bolge_degisimi"][1 - i],
                                  komut=ist["komut"][i].copy())
                out.append(T)
        conn.send(out)


def gae(r, v, g=0.99, l=0.95):
    adv = np.zeros_like(r); son = 0.0
    for t in reversed(range(len(r))):
        vs = v[t + 1] if t + 1 < len(v) else 0.0
        d = r[t] + g * vs - v[t]; son = d + g * l * son; adv[t] = son
    return adv, adv + v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parca", required=True); ap.add_argument("--dunya", required=True)
    ap.add_argument("--alt", default="bulut_sonuc/alt_dovus2_govde2_eppoz6M.pth")
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--mac", type=int, default=2, help="işçi başına maç/güncelleme")
    ap.add_argument("--bc-mac", type=int, default=40); ap.add_argument("--guncelleme", type=int, default=300)
    ap.add_argument("--olcum-her", type=int, default=20); ap.add_argument("--havuz-her", type=int, default=20)
    ap.add_argument("--devam", default=None); ap.add_argument("--onek", default="taktik")
    # 2026-10-22: beceri seçici — düşen robotu kalkış uzmanı (ve yüzüstündeyse yuvarlanma .npz / dönme ağı) kaldırır
    ap.add_argument("--kalkis", default=None); ap.add_argument("--donme", default=None)
    # 2026-10-23: rakip uzaktayken gezinme ağı (taktik_env.gez_yukle; 120 maçta kurala %55 → %62)
    ap.add_argument("--gez", default=None)
    args = ap.parse_args()
    os.chdir(KOK); sys.path.insert(0, KOK)
    import taktik_env as TK
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    ag = Ag(TK.OBS, TK.N_KOMUT); opt = torch.optim.Adam(ag.parameters(), lr=3e-4)
    ctx = mp.get_context("spawn"); conns = []
    for w in range(args.workers):
        a, b = ctx.Pipe(); p = ctx.Process(target=isci, args=(b, w, args), daemon=True); p.start(); conns.append(a)

    def dagit(msgs):
        for c, m in zip(conns, msgs): c.send(m)
        return [x for c in conns for x in c.recv()]

    def olc(n=16):
        n_isci = -(-n // args.workers)
        R = dagit([dict(tur="mac", w=ag.state_dict(), rakip_w=None, n=n_isci, acgozlu=True)] * args.workers)
        sk = [t["bilgi"]["skor"] for t in R]
        return (f"KURALA KARŞI ({len(R)} maç): kazanma %{100 * np.mean([s > 0 for s in sk]):.0f} berabere "
                f"%{100 * np.mean([s == 0 for s in sk]):.0f} | skor ort {np.mean(sk):+.1f} | isabet {np.mean([t['bilgi']['isabet'] for t in R]):.1f} "
                f"alınan {np.mean([t['bilgi']['alinan'] for t in R]):.1f} | bölge değişimi/maç {np.mean([t['bilgi']['bolge'] for t in R]):.2f} | "
                f"komut dağılımı {dict(zip(TK.KOMUTLAR, np.sum([t['bilgi']['komut'] for t in R], 0).tolist()))}")

    if args.devam:
        d = torch.load(args.devam, weights_only=False); ag.load_state_dict(d["ag"]); print(f"devam: {args.devam}", flush=True)
    elif args.bc_mac > 0:
        t0 = time.time()
        V = dagit([dict(tur="bc", n=-(-args.bc_mac // args.workers))] * args.workers)
        X = torch.as_tensor(np.concatenate([x for x, _ in V])); Y = torch.as_tensor(np.concatenate([y for _, y in V]))
        print(f"BC verisi: {len(X):,} karar ({time.time() - t0:.0f} s) | kural komut dağılımı "
              f"{dict(zip(TK.KOMUTLAR, np.bincount(Y.numpy(), minlength=TK.N_KOMUT).tolist()))}", flush=True)
        # 2026-10-22 (ölçüldü): kural kararlarının %4'ü vuruş → ağırlıksız BC %95 doğrulukla neredeyse hiç vurmuyordu
        # (16 maçta 4 vuruş, kurala %31). Sınıf ağırlığı = 1/sıklık (karekök, 10 ile sınırlı).
        say = np.bincount(Y.numpy(), minlength=TK.N_KOMUT).astype(float)
        agirlik = torch.as_tensor(np.clip(np.sqrt(say.max() / np.maximum(say, 1)), 1, 10) * (say > 0), dtype=torch.float32)
        print(f"BC sınıf ağırlıkları: {dict(zip(TK.KOMUTLAR, np.round(agirlik.numpy(), 1).tolist()))}", flush=True)
        for ep in range(30):
            idx = torch.randperm(len(X)); top = 0.0; dog = 0
            for j in range(0, len(X), 512):
                b = idx[j:j + 512]; lg, _ = ag(X[b]); kayip = nn.functional.cross_entropy(lg, Y[b], weight=agirlik)
                opt.zero_grad(); kayip.backward(); opt.step(); top += float(kayip) * len(b); dog += int((lg.argmax(1) == Y[b]).sum())
        print(f"BC bitti: kayıp {top / len(X):.3f}, doğruluk %{100 * dog / len(X):.0f}", flush=True)
        print("BC sonrası " + olc(), flush=True)
    havuz = [{k: v.clone() for k, v in ag.state_dict().items()}]
    for g in range(1, args.guncelleme + 1):
        t0 = time.time()
        msgs = [dict(tur="mac", w=ag.state_dict(), n=args.mac,
                     rakip_w=None if random.random() < 0.5 else random.choice(havuz)) for _ in conns]
        R = dagit(msgs)
        O, A, LP, ADV, RET = [], [], [], [], []
        for t in R:
            adv, ret = gae(t["r"], t["v"]); O.append(t["o"]); A.append(t["a"]); LP.append(t["lp"]); ADV.append(adv); RET.append(ret)
        O, A, LP = torch.as_tensor(np.concatenate(O)), torch.as_tensor(np.concatenate(A)), torch.as_tensor(np.concatenate(LP))
        ADV, RET = torch.as_tensor(np.concatenate(ADV)), torch.as_tensor(np.concatenate(RET))
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
        for _ in range(4):
            idx = torch.randperm(len(O))
            for j in range(0, len(O), 512):
                b = idx[j:j + 512]; lg, v = ag(O[b]); dist = torch.distributions.Categorical(logits=lg)
                oran = torch.exp(dist.log_prob(A[b]) - LP[b])
                kp = -torch.min(oran * ADV[b], torch.clamp(oran, 0.8, 1.2) * ADV[b]).mean()
                kayip = kp + 0.5 * ((v - RET[b]) ** 2).mean() - 0.01 * dist.entropy().mean()
                opt.zero_grad(); kayip.backward(); nn.utils.clip_grad_norm_(ag.parameters(), 0.5); opt.step()
        sk = [t["bilgi"]["skor"] for t in R]
        print(f"g{g:4d} | {len(O):5d} karar {time.time() - t0:4.1f}s | skor ort {np.mean(sk):+5.1f} kazanma %{100 * np.mean([s > 0 for s in sk]):.0f} "
              f"| isabet {np.mean([t['bilgi']['isabet'] for t in R]):.1f} | bölge {np.mean([t['bilgi']['bolge'] for t in R]):.2f}", flush=True)
        if g % args.havuz_her == 0:
            havuz.append({k: v.clone() for k, v in ag.state_dict().items()}); havuz = havuz[-10:]
        if g % args.olcum_her == 0:
            torch.save(dict(ag=ag.state_dict(), g=g), f"{args.onek}_ep{g}.pth")
            print(f"OLCUM g{g}: " + olc(), flush=True)
    torch.save(dict(ag=ag.state_dict(), g=args.guncelleme), f"{args.onek}_son.pth")
    for c in conns: c.send(None)


if __name__ == "__main__":
    main()
