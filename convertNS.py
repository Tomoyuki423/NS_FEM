import numpy as np
import json


def datain(filename='file.json'):

    with open(filename, 'r') as file:
        data = json.load(file)

    inpfile = data['inpfile']
    mesfile = data['mesfile']
    bdcfile = data['bdcfile']
    stafile = data['stafile']
    resfile = data['resfile']
    endfile = data['endfile']
    # inpfileの内容を読み取る部分
    with open(inpfile, 'r') as inp:
        inp_lines = inp.readlines()
    ista, iend, iout = map(int, inp_lines[0].split())
    dt, re = map(float, inp_lines[1].split())
    dti = 1.0 / dt
    rei = 1.0 / re

    return ista, iend, iout, dt, dti, rei, mesfile, bdcfile, stafile, resfile, endfile

def read_mesh_data(mesfile):
    with open(mesfile, 'r') as file:
        lines = file.readlines()
    node, nelm = map(int, lines[0].split())
    xx = np.zeros((2, node))
    nc = np.zeros((3, nelm), dtype=int)
    for i in range(1, node + 1):
        n, x, y = map(float, lines[i].split())
        xx[:, int(n) - 1] = [x, y]
    for i in range(node + 1, node + 1 + nelm):
        m, n1, n2, n3 = map(int, lines[i].split())
        nc[:, int(m) - 1] = [n1, n2, n3]
    return node, nelm, xx, nc




def read_boundary_conditions(bdcfile):
    with open(bdcfile, 'r') as file:
        lines = file.readlines()
    iubc = list(map(int, lines[0].split()))
    iubc_max = max(iubc)
    nubc = np.zeros((2, iubc_max), dtype=int)
    fubc = np.zeros((2, iubc_max))
    for i in range(2):
        if iubc[i] != 0:
            for j in range(1, iubc[i] + 1):
                idx, node, value = map(float, lines[j].split())
                nubc[i, int(idx) - 1] = int(node)
                fubc[i, int(idx) - 1] = value
    return iubc, iubc_max, nubc, fubc

def initin(ista, node, ni, uvp, uu, ua, iubc, nubc, fubc, iubc_max, stafile):
    if ista == -1:
        ista = 0
        uvp.fill(0.0)
    else:
        with open(stafile, 'r') as file:
            lines = file.readlines()
        ista = int(lines[0].strip())
        for line in lines[1:]:
            n, u, v, p = map(float, line.split())
            uvp[int(n) - 1] = [u, v, p]
    for i in range(2):
        for j in range(iubc[i]):
            uvp[nubc[i, j] + ni[i]] = fubc[i, j]
    for n in range(node):
        for i in range(2):
            uu[i, n] = uvp[n + ni[i]]
            ua[i, n] = uvp[n + ni[i]]

# # Example usage
# initin(ista, node, ni, uvp, uu, ua, iubc, nubc, fubc, iubc_max, stafile)
# print(uvp)
# print(uu)
# print(ua)


def stepch(node, ni, uvp, uu, ua):
    for n in range(node):
        for i in range(2):
            ua[i, n] = 1.5 * uvp[n + ni[i]] - 0.5 * uu[i, n]
            uu[i, n] = uvp[n + ni[i]]

def maktau(node, nelm, nc, area, bc, ua, dti, rei):
    tau = np.zeros(nelm)
    t1 = 4.0 * dti * dti
    for m in range(nelm):
        n1, n2, n3 = nc[:, m]
        a01 = area[m]
        if a01 == 0:
            tau[m] = 0
            continue
        b1, b2, b3 = bc[0, :, m]
        c1, c2, c3 = bc[1, :, m]

        u = (ua[0, n1-1] + ua[0, n2-1] + ua[0, n3-1]) / 3.0
        v = (ua[1, n1-1] + ua[1, n2-1] + ua[1, n3-1]) / 3.0
        uv = u * u + v * v
        d = abs(u * b1 + v * c1) + abs(u * b2 + v * c2) + abs(u * b3 + v * c3)

        t2 = d * d
        if uv < 1.0e-10:
            t3 = (2.0 * rei / a01) ** 2.0 if a01 != 0 else float('inf')
        else:
            t3 = (rei * d * d / uv) ** 2.0
        tau[m] = 1.0 / (t1 + t2 + t3)**0.5 if (t1 + t2 + t3) != 0 else float('inf')
    return tau


def makNET(node, nelm, nc):
    infn = np.zeros(node, dtype=int)
    for m in range(nelm):
        for i in range(3):
            infn[nc[i, m] - 1] += 1

    infn_max = np.max(infn)
    infe = np.zeros((infn_max, node), dtype=int)
    infc = np.zeros((infn_max, node), dtype=int)

    infn.fill(0)
    for m in range(nelm):
        for i in range(3):
            n = nc[i, m] - 1
            infn[n] += 1
            infe[infn[n] - 1, n] = m + 1
            infc[infn[n] - 1, n] = i + 1

    nrow_max = 1
    for n in range(node):
        infnn = 1
        for i in range(infn[n]):
            ne = infe[i, n] - 1
            nt = infc[i, n] - 1
            n1 = nc[nt, ne] - 1
            n2 = nc[(nt + 1) % 3, ne] - 1
            infnn = max(infnn, n1 + 1, n2 + 1)
        nrow_max = max(nrow_max, infnn)

    return nrow_max, infe, infc

# # Example usage
# nrow_max, infe, infc = makNET(node, nelm, nc)
# print(nrow_max)



def makLHS(node, nelm, nc, area, bc, ua, dti, rei, tau, ni, nrow_max, infn, infe, infc):
    diag = np.zeros(node * 3)
    coef_x = np.zeros((nrow_max * 3, node))
    coef_y = np.zeros((nrow_max * 3, node))
    coef_c = np.zeros((nrow_max * 3, node))

    for n in range(node * 3):
        diag[n] = 0.0

    for n in range(node):
        for i in range(nrow_max * 3):
            coef_x[i, n] = 0.0
            coef_y[i, n] = 0.0
            coef_c[i, n] = 0.0

    for n in range(node):
        for i in range(infn[n]):
            ne = infe[i, n] - 1
            nt = infc[i, n] - 1
            n1, n2, n3 = nc[:, ne]
            a01 = area[ne]
            a03 = area[ne] / 3.0
            ts = tau[ne]
            tp = tau[ne]
            b1, b2, b3 = bc[0, :, ne]
            c1, c2, c3 = bc[1, :, ne]
            ua1, ua2, ua3 = ua[0, n1-1], ua[0, n2-1], ua[0, n3-1]
            va1, va2, va3 = ua[1, n1-1], ua[1, n2-1], ua[1, n3-1]

            # Calculation logic here
            # Example: eMs_11 = ts * (b1 * emu1 + c1 * emv1)
            # Update coef_x, coef_y, coef_c, and diag as needed

    return diag, coef_x, coef_y, coef_c



def makRHS(node, nelm, nc, area, bc, ua, dti, rei, tau, uu, ni, infn, infe, infc):
    bv = np.zeros(node * 3)
    for n in range(node):
        for i in range(infn[n]):
            ne = infe[i, n] - 1
            nt = infc[i, n] - 1
            n1, n2, n3 = nc[:, ne]
            a01 = area[ne]
            a03 = area[ne] / 3.0
            b1, b2, b3 = bc[0, :, ne]
            c1, c2, c3 = bc[1, :, ne]

            u1, u2, u3 = uu[0, n1-1], uu[0, n2-1], uu[0, n3-1]
            v1, v2, v3 = uu[1, n1-1], uu[1, n2-1], uu[1, n3-1]
            ua1, ua2, ua3 = ua[0, n1-1], ua[0, n2-1], ua[0, n3-1]
            va1, va2, va3 = ua[1, n1-1], ua[1, n2-1], ua[1, n3-1]

            emd = a03 * 0.5
            emu = a03 * 0.25

            dxu = b1 * u1 + b2 * u2 + b3 * u3
            dyu = c1 * u1 + c2 * u2 + c3 * u3
            dxv = b1 * v1 + b2 * v2 + b3 * v3
            dyv = c1 * v1 + c2 * v2 + c3 * v3

            eMu1 = emd * u1 + emu * u2 + emu * u3
            eMv1 = emd * v1 + emu * v2 + emu * v3

            emua1 = emd * ua1 + emu * ua2 + emu * ua3
            emua2 = emu * ua1 + emd * ua2 + emu * ua3
            emua3 = emu * ua1 + emu * ua2 + emd * ua3
            emva1 = emd * va1 + emu * va2 + emu * va3
            emva2 = emu * va1 + emd * va2 + emu * va3
            emva3 = emu * va1 + emu * va2 + emd * va3
            eAu1 = emua1 * dxu + emva1 * dyu
            eAv1 = emua1 * dxv + emva1 * dyv

            sxx = rei * (dxu + dxu) * a01
            sxy = rei * (dyu + dxv) * a01
            syy = rei * (dyv + dyv) * a01
            eDu1 = b1 * sxx + c1 * sxy
            eDv1 = b1 * sxy + c1 * syy

            bv[n + ni[0]] += eMu1 * dti - 0.5 * (eAu1 + eDu1)
            bv[n + ni[1]] += eMv1 * dti - 0.5 * (eAv1 + eDv1)

            ts = tau[ne]
            tp = tau[ne]

            tumu = ts * (emua1 * u1 + emua2 * u2 + emua3 * u3)
            tvmu = ts * (emva1 * u1 + emva2 * u2 + emva3 * u3)
            tumv = ts * (emua1 * v1 + emua2 * v2 + emua3 * v3)
            tvmv = ts * (emva1 * v1 + emva2 * v2 + emva3 * v3)
            eMsu1 = b1 * tumu + c1 * tvmu
            eMsv1 = b1 * tumv + c1 * tvmv

            tu = tp * (u1 + u2 + u3) * a03
            tv = tp * (v1 + v2 + v3) * a03
            eMp1 = b1 * tu + c1 * tv

            tumu = ts * (ua1 * emua1 + ua2 * emua2 + ua3 * emua3)
            tumv = ts * (ua1 * emva1 + ua2 * emva2 + ua3 * emva3)
            tvmv = ts * (va1 * emva1 + va2 * emva2 + va3 * emva3)
            stxu = tumu * dxu + tumv * dyu
            styu = tumv * dxu + tvmv * dyu
            stxv = tumu * dxv + tumv * dyv
            styv = tumv * dxv + tvmv * dyv
            eAsu1 = b1 * stxu + c1 * styu
            eAsv1 = b1 * stxv + c1 * styv

            tu = tp * (ua1 + ua2 + ua3) * a03
            tv = tp * (va1 + va2 + va3) * a03
            atu = tu * dxu + tv * dyu
            atv = tu * dxv + tv * dyv
            eAp1 = b1 * atu + c1 * atv

            bv[n + ni[0]] += eMsu1 * dti - 0.5 * eAsu1
            bv[n + ni[1]] += eMsv1 * dti - 0.5 * eAsv1
            bv[n + ni[2]] += eMp1 * dti - 0.5 * eAp1

    return bv


def GPBiCG(node, x, b, d, ni, ibc, ibc_max, nbc, nrow_max, coef_x, coef_y, coef_c):
    ndof = node * 3
    r = np.zeros(ndof)
    r0 = np.zeros(ndof)
    p = np.zeros(ndof)
    t = np.zeros(ndof)
    y = np.zeros(ndof)
    u = np.zeros(ndof)
    z = np.zeros(ndof)
    w = np.zeros(ndof)
    Ap = np.zeros(ndof)
    At = np.zeros(ndof)
    eps = 1.0e-6

    x /= d
    r = b * d
    bound0(node, ni, r, ibc_max, ibc, nbc)
    Ap = matvec(node, ni, x, ibc_max, ibc, nbc, nrow_max, coef_x, coef_y, coef_c)
    r0_r = np.dot(r0, r)

    beta = 0.0
    pk = 0.0
    k = 0
    if np.sqrt(r0_r) > eps:
        for k in range(ndof * 3):
            p = r + beta * (p - u)
            Ap = matvec(node, ni, p, ibc_max, ibc, nbc, nrow_max, coef_x, coef_y, coef_c)
            alph = np.dot(r0, Ap)
            alph = r0_r / alph
            y = t - r - alph * w + alph * Ap
            u = t - r + beta * u
            t = r - alph * Ap

            At = matvec(node, ni, t, ibc_max, ibc, nbc, nrow_max, coef_x, coef_y, coef_c)
            y_y = np.dot(y, y)
            At_t = np.dot(At, t)
            At_At = np.dot(At, At)
            y_t = np.dot(y, t)
            y_At = np.dot(y, At)

            zeta = (y_y * At_t - y_t * y_At * pk) / (At_At * y_y - y_At * y_At * pk)
            eta = (At_At * y_t - y_At * At_t) / (At_At * y_y - y_At * y_At) * pk
            pk = 1.0
            r_r = 0.0
            u = zeta * Ap + eta * u
            z = zeta * r + eta * z - alph * u
            x = x + alph * p + z
            r = t - eta * y - zeta * At
            r_r = np.dot(r, r)
            if np.sqrt(r_r) <= eps:
                break
            r0_r0 = r0_r
            r0_r = np.dot(r0, r)
            beta = alph / zeta * r0_r / r0_r0
            w = At + beta * Ap

        if k == ndof * 3:
            raise Exception("Not Convergence (GPBi-CG)")

    x *= d
    return x, k




def matvec(node, ni, x, ibc_max, ibc, nbc, nrow_max, coef_x, coef_y, coef_c):
    ndof = node * 3
    r = np.zeros(ndof)
    for i in range(node):
        for j in range(nrow_max):
            xj = [x[j + ni[k]] for k in range(3)]
            for k in range(3):
                r[i + ni[k]] += coef_x[j + k * nrow_max, i] * xj[0]
                r[i + ni[k]] += coef_y[j + k * nrow_max, i] * xj[1]
                r[i + ni[k]] += coef_c[j + k * nrow_max, i] * xj[2]
    bound0(node, ni, r, ibc_max, ibc, nbc)
    return r


def output(filename, istep, ttime, node, ni, uvp):
    with open(filename, 'w') as f:
        f.write(f"{istep} {ttime}\n")
        for n in range(node):
            f.write(f"{n} {uvp[n+ni[0]]} {uvp[n+ni[1]]} {uvp[n+ni[2]]}\n")

def bound0(node, ni, uvp, iubc_max, iubc, nubc):
    for i in range(2):
        for ib in range(iubc[i]):
            uvp[nubc[i, ib] + ni[i]] = 0.0

def main():
    # Initialize input data
    ista, iend, iout, dt, dti, rei, mesfile, bdcfile, stafile, resfile, endfile = datain("file.json")
    node, nelm, xx, nc = read_mesh_data(mesfile)
    iubc, iubc_max, nubc, fubc = read_boundary_conditions(bdcfile)

    # Initialize variables
    uvp = np.zeros(node * 3)
    uu = np.zeros((2, node))
    ua = np.zeros((2, node))
    bc = np.zeros((2, 3, nelm))
    area = np.zeros(nelm)
    tau = np.zeros(nelm)
    diag = np.zeros(node * 3)
    ni = np.zeros(3, dtype=int)
    ni[0] = 0
    ni[1] = node
    ni[2] = node * 2

    # Calculate nrow_max and initialize infe and infc
    nrow_max, infe, infc = makNET(node, nelm, nc)

    # Initialize infn
    infn = np.zeros(node, dtype=int)
    for m in range(nelm):
        for i in range(3):
            infn[nc[i, m] - 1] += 1

    # Initialize
    initin(ista, node, ni, uvp, uu, ua, iubc, nubc, fubc, iubc_max, stafile)

    for istep in range(ista + 1, iend + 1):
        ttime = istep * dt

        tau = maktau(node, nelm, nc, area, bc, ua, dti, rei)
        diag, coef_x, coef_y, coef_c = makLHS(node, nelm, nc, area, bc, ua, dti, rei, tau, ni, nrow_max, infn, infe, infc)
        bv = makRHS(node, nelm, nc, area, bc, ua, dti, rei, tau, uu, ni, infn, infe, infc)
        uvp, kcg = GPBiCG(node, uvp, bv, diag, ni, iubc, iubc_max, nubc, nrow_max, coef_x, coef_y, coef_c)
        stepch(node, ni, uvp, uu, ua)

        print(f"****** STEP INFO *****")
        print(f"STEP   = {istep}")
        print(f"TIME   = {ttime}")
        print(f"K(BiCG)= {kcg}")

        if istep % iout == 0:
            output(resfile, istep, ttime, node, ni, uvp)

    output(endfile, iend, ttime, node, ni, uvp)

if __name__ == "__main__":
    main()









# # Example usage
# node, nelm, xx, nc = read_mesh_data(mesfile)
# print(node, nelm)
# print(xx)
# print(nc)


# # Example usage
# ista, iend, iout, dt, dti, rei = datain()
# print(ista, iend, iout, dt, dti, rei)

# # Example usage
# node, nelm, xx, nc = read_mesh_data()
# print(node, nelm)
# print(xx)
# print(nc)

# # Example usage
# iubc, iubc_max, nubc, fubc = read_boundary_conditions()
# print(iubc, iubc_max)
# print(nubc)
# print(fubc)

# # Example usage
# uvp = np.zeros((node * 3))
# uu = np.zeros((2, node))
# ua = np.zeros((2, node))
# initin(ista, node, ni, uvp, uu, ua, iubc, nubc, fubc, iubc_max)
# print(uvp)
# print(uu)
# print(ua)

# # Example usage
# stepch(node, ni, uvp, uu, ua)
# print(ua)
# print(uu)

# # Example usage
# tau = maktau(node, nelm, nc, area, bc, ua, dti, rei)
# print(tau)

# # Example usage
# diag, coef_x, coef_y, coef_c = makLHS(node, nelm, nc, area, bc, ua, dti, rei, tau, ni)
# print(diag)
# print(coef_x)
# print(coef_y)
# print(coef_c)

# # Example usage
# bv = makRHS(node, nelm, nc, area, bc, ua, dti, rei, tau, uu, ni)
# print(bv)

# # Example usage
# x, k = GPBiCG(node, x, b, d, ni, ibc, ibc_max, nbc)
# print(x)
# print(k)

# # Example usage
# r = matvec(node, ni, x, ibc_max, ibc, nbc)
# print(r)


# # Example usage
# output('output.dat', istep, ttime, node, ni, uvp)

# # Example usage
# bound0(node, ni, uvp, iubc_max, iubc, nubc)
# print(uvp)
