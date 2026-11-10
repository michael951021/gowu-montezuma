import signal


def worker_x(conn, seeds):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    from train_route import make_env
    from env.ram_features import read_progress     # diagnostics only
    envs = [make_env(s) for s in seeds]
    steps = [0] * len(envs)
    conn.send([e.reset()[0] for e in envs])
    try:
        while True:
            msg = conn.recv()
            if msg is None:
                break
            out = []
            for j, (e, (a, cap)) in enumerate(zip(envs, msg)):
                obs, r, term, trunc, _ = e.step(int(a)); steps[j] += 1
                p = read_progress(e.unwrapped.ale)
                done = term or trunc or steps[j] > cap
                if done:
                    obs, _ = e.reset(); steps[j] = 0
                out.append((obs, float(r), done, int(p.lives), int(p.room), int(p.x), int(p.y)))
            conn.send(out)
    except (EOFError, KeyboardInterrupt):
        pass
