#!/usr/bin/env python3
"""
乱流壁法則 理論解 vs Ansys Fluent CFD 比較
============================================
2次元チャネル乱流の壁法則を k-ω SST モデル (壁面解像メッシュ) でシミュレーションし、
理論解 (粘性底層・対数則) と比較する。

理論:
  粘性底層 : u⁺ = y⁺                       (y⁺ < 5)
  対数則   : u⁺ = (1/κ) ln(y⁺) + B         (y⁺ > 30)
             κ = 0.41 (von Kármán 定数), B = 5.0

使用法:
  python wall_function_comparison.py            # フル実行 (メッシュ生成→CFD→プロット)
  python wall_function_comparison.py --mesh-only # メッシュ生成のみ
  python wall_function_comparison.py --plot-only # 既存データからプロットのみ
"""

import os
import sys
import argparse
import subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ─── Ansys / Fluent パス ─────────────────────────────────────────────────────
ANSYS_ROOT   = "/home/opt/ansys/v252"
FLUENT_ROOT  = os.path.join(ANSYS_ROOT, "fluent")
FLUENT_BIN   = os.path.join(FLUENT_ROOT, "bin", "fluent")

os.environ["PYFLUENT_FLUENT_ROOT"] = FLUENT_ROOT  # PyFluent バージョン制限を回避
os.environ["AWP_ROOT252"]          = ANSYS_ROOT

# ─── 作業ディレクトリ ─────────────────────────────────────────────────────────
WORK_DIR          = os.path.dirname(os.path.abspath(__file__))
MESH_FILE         = os.path.join(WORK_DIR, "channel.msh")
JOURNAL_FILE      = os.path.join(WORK_DIR, "setup.jou")
VELOCITY_FILE     = os.path.join(WORK_DIR, "velocity_profile.dat")
WALL_SHEAR_FILE   = os.path.join(WORK_DIR, "wall_shear.dat")
OUTPUT_PNG        = os.path.join(WORK_DIR, "wall_function_comparison.png")

# ─── 流体物性 (空気, 常温常圧) ────────────────────────────────────────────────
RHO = 1.225      # kg/m³  密度
MU  = 1.789e-5   # Pa·s   粘性係数
NU  = MU / RHO   # m²/s   動粘性係数

# ─── 計算ドメイン ─────────────────────────────────────────────────────────────
CHANNEL_LENGTH = 1.0   # m  流路長
CHANNEL_HEIGHT = 0.1   # m  流路高さ (全高)
U_BULK         = 5.0   # m/s 平均流速

# 見込み摩擦速度 (Blasius 相関で事前推定)
RE_D   = U_BULK * CHANNEL_HEIGHT / NU
_f_est = 0.316 * RE_D**(-0.25)            # Blasius 摩擦係数
_TAU_W = (_f_est / 8.0) * RHO * U_BULK**2
U_TAU_EST = (_TAU_W / RHO)**0.5           # ≈ 0.27 m/s

# ─── メッシュパラメータ ────────────────────────────────────────────────────────
# 第 1 セル中心 y⁺ ≈ 1 (k-ω SST 壁面解像): 幾何級数 y 間隔
NX      = 100    # x 方向 (一様)
NY_HALF = 25     # y 方向 半チャネル分のセル数 (幾何級数, 壁から離れるにつれ拡大)
NY      = NY_HALF * 2   # y 方向合計セル数 (上下対称)
R_EXP   = 1.2    # y 方向幾何拡大率
# dy1 = (H/2)*(r-1)/(r^N-1) → 第1セル中心 y⁺ ≈ 1 になるよう設計
_DY1    = (CHANNEL_HEIGHT / 2.0) * (R_EXP - 1.0) / (R_EXP**NY_HALF - 1.0)

# ─── 壁法則定数 ───────────────────────────────────────────────────────────────
KAPPA = 0.41
B_LOG = 5.0

# ─── 日本語フォント設定 ───────────────────────────────────────────────────────
def _setup_japanese_font():
    """利用可能な日本語フォントを探して設定する"""
    candidates = [
        "Noto Sans CJK JP", "IPAexGothic", "IPAPGothic", "VL Gothic",
        "Takao Gothic", "M+ 1p", "DejaVu Sans",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            matplotlib.rcParams["font.family"] = name
            return name
    # フォールバック: 英語ラベルで描画
    return None

# ═══════════════════════════════════════════════════════════════════════════════
# 1. 理論解
# ═══════════════════════════════════════════════════════════════════════════════

def u_plus_theory(y_plus: np.ndarray) -> np.ndarray:
    """壁法則の複合式 (粘性底層 / バッファ層補間 / 対数則)"""
    yp = np.asarray(y_plus, dtype=float)
    up = np.empty_like(yp)

    visc = yp < 5.0
    log  = yp >= 30.0
    buf  = ~visc & ~log

    up[visc] = yp[visc]
    up[log]  = (1.0 / KAPPA) * np.log(yp[log]) + B_LOG

    # バッファ層: 粘性底層と対数則の線形補間
    alpha       = (yp[buf] - 5.0) / 25.0
    up_visc_end = 5.0
    up_log_start = (1.0 / KAPPA) * np.log(30.0) + B_LOG
    up[buf] = (1.0 - alpha) * up_visc_end + alpha * up_log_start
    return up


# ═══════════════════════════════════════════════════════════════════════════════
# 2. メッシュ生成 (Fluent ASCII .msh フォーマット)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_channel_mesh(filename: str, Nx: int, Ny: int,
                          L: float, H: float) -> None:
    """
    2D 構造化四角形メッシュを Fluent ASCII .msh 形式で出力する。

    ゾーン構成:
      zone 1: fluid  (セル領域)
      zone 2: inlet  (velocity-inlet,  x=0)
      zone 3: outlet (pressure-outlet, x=L)
      zone 4: bottom-wall (wall, y=0)
      zone 5: top-wall    (wall, y=H)
      zone 6: interior    (内部面)
    """
    x_nodes = np.linspace(0.0, L, Nx + 1)

    # ── 幾何級数 y 節点 (両壁面で細かく、チャネル中央で粗い対称メッシュ) ──
    half_H = H / 2.0
    dy1 = half_H * (R_EXP - 1.0) / (R_EXP**NY_HALF - 1.0)
    y_half = np.zeros(NY_HALF + 1)
    dy = dy1
    for k in range(NY_HALF):
        y_half[k + 1] = y_half[k] + dy
        dy *= R_EXP
    y_nodes = np.concatenate([y_half, H - y_half[-2::-1]])

    n_nodes   = (Nx + 1) * (Ny + 1)
    n_cells   = Nx * Ny
    n_inlet   = Ny
    n_outlet  = Ny
    n_bot     = Nx
    n_top     = Nx
    n_int_h   = Nx * (Ny - 1)        # 水平内部面
    n_int_v   = (Nx - 1) * Ny        # 垂直内部面
    n_faces   = n_inlet + n_outlet + n_bot + n_top + n_int_h + n_int_v

    def nid(i, j):  # 1-based ノード番号
        return j * (Nx + 1) + i + 1

    def cid(i, j):  # 1-based セル番号
        return j * Nx + i + 1

    def h(n):       # 10進数→16進数文字列 (小文字)
        return format(n, "x")

    with open(filename, "w") as f:
        f.write('(0 "Fluent 2D Channel Mesh for Wall-Function Test")\n\n')
        f.write("(2 2)\n\n")  # 2次元

        # ── ノード ──────────────────────────────────────────────────────────
        f.write(f"(10 (0 1 {h(n_nodes)} 0 2))\n")
        f.write(f"(10 (1 1 {h(n_nodes)} 1 2)\n(\n")
        for j in range(Ny + 1):
            for i in range(Nx + 1):
                f.write(f"  {x_nodes[i]:.10e} {y_nodes[j]:.10e}\n")
        f.write("))\n\n")

        # ── セル (四角形 = 3) ───────────────────────────────────────────────
        f.write(f"(12 (0 1 {h(n_cells)} 0 0))\n")
        f.write(f"(12 (1 1 {h(n_cells)} 1 3))\n\n")

        # ── 面ヘッダー ───────────────────────────────────────────────────────
        f.write(f"(13 (0 1 {h(n_faces)} 0 0))\n\n")

        fid = 1  # 現在の面インデックス

        # ── Zone 2: inlet (velocity-inlet = 0xa) ────────────────────────────
        # エッジ (0,j+1)→(0,j) : 法線 = +x → c0=セル(0,j), c1=0(境界)
        f1, fl = fid, fid + n_inlet - 1
        f.write(f"(13 ({h(2)} {h(f1)} {h(fl)} a 2)\n(\n")
        for j in range(Ny):
            f.write(f" {h(nid(0,j+1))} {h(nid(0,j))} {h(cid(0,j))} 0\n")
        f.write("))\n\n")
        fid += n_inlet

        # ── Zone 3: outlet (pressure-outlet = 0x5) ──────────────────────────
        # エッジ (Nx,j)→(Nx,j+1) : 法線 = -x → c0=セル(Nx-1,j), c1=0
        f1, fl = fid, fid + n_outlet - 1
        f.write(f"(13 ({h(3)} {h(f1)} {h(fl)} 5 2)\n(\n")
        for j in range(Ny):
            f.write(f" {h(nid(Nx,j))} {h(nid(Nx,j+1))} {h(cid(Nx-1,j))} 0\n")
        f.write("))\n\n")
        fid += n_outlet

        # ── Zone 4: bottom-wall (wall = 0x3, y=0) ───────────────────────────
        # エッジ (i,0)→(i+1,0) : 法線 = +y → c0=セル(i,0), c1=0
        f1, fl = fid, fid + n_bot - 1
        f.write(f"(13 ({h(4)} {h(f1)} {h(fl)} 3 2)\n(\n")
        for i in range(Nx):
            f.write(f" {h(nid(i,0))} {h(nid(i+1,0))} {h(cid(i,0))} 0\n")
        f.write("))\n\n")
        fid += n_bot

        # ── Zone 5: top-wall (wall = 0x3, y=H) ──────────────────────────────
        # エッジ (i+1,Ny)→(i,Ny) : 法線 = -y → c0=セル(i,Ny-1), c1=0
        f1, fl = fid, fid + n_top - 1
        f.write(f"(13 ({h(5)} {h(f1)} {h(fl)} 3 2)\n(\n")
        for i in range(Nx):
            f.write(f" {h(nid(i+1,Ny))} {h(nid(i,Ny))} {h(cid(i,Ny-1))} 0\n")
        f.write("))\n\n")
        fid += n_top

        # ── Zone 6: interior (interior = 0x2) ───────────────────────────────
        f1, fl = fid, fid + n_int_h + n_int_v - 1
        f.write(f"(13 ({h(6)} {h(f1)} {h(fl)} 2 2)\n(\n")

        # 水平内部面 (j=1..Ny-1)
        # エッジ (i,j)→(i+1,j) : 法線 = +y → c0=上セル(i,j), c1=下セル(i,j-1)
        for j in range(1, Ny):
            for i in range(Nx):
                f.write(f" {h(nid(i,j))} {h(nid(i+1,j))} {h(cid(i,j))} {h(cid(i,j-1))}\n")

        # 垂直内部面 (i=1..Nx-1)
        # エッジ (i,j)→(i,j+1) : 法線 = -x → c0=左セル(i-1,j), c1=右セル(i,j)
        for i in range(1, Nx):
            for j in range(Ny):
                f.write(f" {h(nid(i,j))} {h(nid(i,j+1))} {h(cid(i-1,j))} {h(cid(i,j))}\n")

        f.write("))\n\n")

        # ── ゾーン名 (正式フォーマット: zone-id zone-type zone-name) ──────────
        f.write("(45 (1 fluid fluid)())\n")
        f.write("(45 (2 velocity-inlet inlet)())\n")
        f.write("(45 (3 pressure-outlet outlet)())\n")
        f.write("(45 (4 wall bottom-wall)())\n")
        f.write("(45 (5 wall top-wall)())\n")
        f.write("(45 (6 interior interior-6)())\n")

    print(f"[mesh] 生成完了: {filename}")
    print(f"       ノード {n_nodes:,}, セル {n_cells:,}, 面 {n_faces:,}")
    dy1_mm = dy1 * 1e3
    yp1_est = (dy1 / 2.0) * U_TAU_EST / NU   # 第1セル中心の推定 y⁺
    print(f"       第1セル高さ dy1 = {dy1_mm:.4f} mm, 推定 y⁺(第1セル中心) ≈ {yp1_est:.2f}")
    print(f"       拡大率 R = {R_EXP}, 半チャネル分 NY_HALF = {NY_HALF}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Fluent ジャーナルファイル生成
# ═══════════════════════════════════════════════════════════════════════════════

def write_fluent_journal(journal_file: str, mesh_file: str,
                         vel_out: str, shear_out: str) -> None:
    """
    Fluent 25.2 TUI ジャーナルを生成する。
    k-ω SST モデル (壁面解像, y⁺≈1), SIMPLE 法。
    各プロンプトへの応答は Fluent 25.2 で実際に確認済み。
    """
    x_mid = CHANNEL_LENGTH / 2.0
    Dh    = 2.0 * CHANNEL_HEIGHT  # 水力直径 (2D 無限スパン近似)

    jou = f"""; ================================================================
; Fluent Journal: 2D Channel Flow - Wall Function Comparison
; Fluent 25.2  /  k-omega SST (wall-resolved, y+~1)
; ================================================================

; メッシュ読み込み
/file/read-case "{mesh_file}"

; ── 乱流モデル: k-ω SST (壁面解像) ───────────────────────────
/define/models/viscous/kw-sst yes

; ── 境界条件: 速度入口 ────────────────────────────────────────
; Velocity Specification Method: Magnitude and Direction [no] -> no
; Velocity Specification Method: Components [no]              -> no
; Velocity Specification Method: Magnitude, Normal to Boundary [yes] -> yes
; Reference Frame: Absolute [yes]                             -> yes
; Use Profile for Velocity Magnitude? [no]                    -> no
; Velocity Magnitude [m/s] [0]                                -> {U_BULK}
; Use Profile for Supersonic/Initial Gauge Pressure? [no]     -> no
; Supersonic/Initial Gauge Pressure [Pa] [0]                  -> 0
; Turbulence: K and Omega [no]                                -> no
; Turbulence: Intensity and Length Scale [no]                 -> no
; Turbulence: Intensity and Viscosity Ratio [yes]             -> no
; Turbulence: Intensity and Hydraulic Diameter [no]           -> yes
; Turbulent Intensity [%] [5]                                 -> 5
; Hydraulic Diameter [m] [1]                                  -> {Dh}
/define/boundary-conditions/velocity-inlet
inlet
no
no
yes
yes
no
{U_BULK}
no
0
no
no
no
yes
5
{Dh}

; ── 境界条件: 圧力出口 ───────────────────────────────────────
; Backflow Reference Frame: Absolute [yes]                    -> yes
; Use Profile for Gauge Pressure? [no]                        -> no
; Gauge Pressure [Pa] [0]                                     -> 0
; Backflow Direction: Direction Vector [no]                   -> no
; Backflow Direction: Normal to Boundary [yes]                -> yes
; Turbulence: K and Omega [no]                                -> no
; Turbulence: Intensity and Length Scale [no]                 -> no
; Turbulence: Intensity and Viscosity Ratio [yes]             -> no
; Turbulence: Intensity and Hydraulic Diameter [no]           -> yes
; Backflow Turbulent Intensity [%] [5]                        -> 5
; Backflow Hydraulic Diameter [m] [1]                         -> {Dh}
; Backflow Pressure Specification: Total Pressure [yes]       -> yes
; Average Pressure Specification? [no]                        -> no
; Specify targeted mass flow rate [no]                        -> no
/define/boundary-conditions/pressure-outlet
outlet
yes
no
0
no
yes
no
no
no
yes
5
{Dh}
yes
no
no

; ── 初期化 & 計算 ─────────────────────────────────────────────
/solve/initialize/initialize-flow
/solve/iterate 500

; ── ポスト処理 ───────────────────────────────────────────────
; x = L/2 の縦断面ライン
/surface/line-surface
centerline
{x_mid} 0
{x_mid} {CHANNEL_HEIGHT / 2.0}

; 速度プロファイル出力 (nodenumber x y x-velocity の順)
/file/export/ascii "{vel_out}" centerline () no x-velocity () no

; 壁面せん断応力 (面積加重平均) 出力
/report/surface-integrals/area-weighted-avg
bottom-wall
()
wall-shear
yes
"{shear_out}"

/exit yes
"""
    with open(journal_file, "w") as f:
        f.write(jou)
    print(f"[journal] 生成完了: {journal_file}")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. PyFluent によるシミュレーション実行
# ═══════════════════════════════════════════════════════════════════════════════

def run_fluent_simulation() -> None:
    """Fluent を直接起動し、ジャーナルを実行する。"""
    # 出力ファイルが既に存在する場合は削除 (Fluent の "Append?" プロンプト回避)
    for fpath in [VELOCITY_FILE, WALL_SHEAR_FILE]:
        if os.path.exists(fpath):
            os.remove(fpath)

    print("\n[fluent] Fluent 25.2 を起動中...")
    cmd = [FLUENT_BIN, "2ddp", "-t64", "-gu", "-i", JOURNAL_FILE]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        print("[fluent] stderr:", result.stderr[-2000:] if result.stderr else "(なし)")

    # 出力ファイルの存在確認
    if not os.path.exists(VELOCITY_FILE):
        raise RuntimeError(f"速度プロファイルファイルが見つかりません: {VELOCITY_FILE}")
    if not os.path.exists(WALL_SHEAR_FILE):
        raise RuntimeError(f"壁面せん断応力ファイルが見つかりません: {WALL_SHEAR_FILE}")

    print("[fluent] シミュレーション完了")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. データ読み込み・解析
# ═══════════════════════════════════════════════════════════════════════════════

def read_velocity_profile(filepath: str) -> np.ndarray:
    """
    /file/export/ascii で書き出した速度プロファイルを読み込む。
    ヘッダ行: nodenumber x-coordinate y-coordinate x-velocity ...
    Returns: shape (N, 2) — [y, u_x]  (y=0 の壁面点を除く)
    """
    rows = []
    with open(filepath) as f:
        header_skipped = False
        for line in f:
            line = line.strip()
            if not line:
                continue
            if not header_skipped:
                header_skipped = True
                continue  # 最初の行はヘッダ
            try:
                vals = [float(v) for v in line.split()]
                # 列: nodenumber, x, y, x-velocity, ...
                if len(vals) >= 4:
                    y_val = vals[2]
                    u_val = vals[3]
                    if y_val > 1e-12:  # 壁面点 (y=0) を除く
                        rows.append([y_val, u_val])
            except ValueError:
                continue
    if not rows:
        raise RuntimeError(f"速度プロファイルが読み込めませんでした: {filepath}")
    data = np.array(rows)
    data = data[data[:, 0].argsort()]
    return data


def read_wall_shear(filepath: str) -> float:
    """
    /report/surface-integrals/area-weighted-avg で書き出した
    壁面せん断応力レポートから数値を読み込む。
    Returns: τ_w (Pa) as float
    """
    with open(filepath) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    val = float(parts[-1])
                    if val > 0:
                        tau_w = val
                except ValueError:
                    continue
    try:
        return tau_w
    except NameError:
        raise RuntimeError(f"壁面せん断応力が読み込めませんでした: {filepath}")


def compute_wall_units(profile: np.ndarray, tau_w: float):
    """
    CFD データから壁単位 (y⁺, u⁺) を計算する。

    Parameters
    ----------
    profile : shape (N, 2) — [y, u_x]  (y=0 の壁面点を除く)
    tau_w   : float — 壁面せん断応力 [Pa]

    Returns
    -------
    y_plus, u_plus, u_tau
    """
    u_tau = np.sqrt(tau_w / RHO)

    y = profile[:, 0]
    u = profile[:, 1]

    y_plus = y * u_tau / NU
    u_plus = u / u_tau

    return y_plus, u_plus, u_tau


# ═══════════════════════════════════════════════════════════════════════════════
# 6. プロット
# ═══════════════════════════════════════════════════════════════════════════════

def plot_comparison() -> str:
    """理論解と CFD 結果を比較する u⁺–y⁺ グラフを生成する。"""
    font_name = _setup_japanese_font()
    use_jp = font_name is not None

    # ── データ読み込み ─────────────────────────────────────────────────────
    profile = read_velocity_profile(VELOCITY_FILE)
    tau_w   = read_wall_shear(WALL_SHEAR_FILE)
    y_plus_cfd, u_plus_cfd, u_tau = compute_wall_units(profile, tau_w)

    Re_tau = int(u_tau * (CHANNEL_HEIGHT / 2.0) / NU)
    Re_D   = int(U_BULK * CHANNEL_HEIGHT / NU)
    print(f"\n[result] τ_w = {tau_w:.4f} Pa | u_τ = {u_tau:.4f} m/s | "
          f"Re_τ ≈ {Re_tau} | Re_D = {Re_D:,}")

    # ── 理論曲線 ──────────────────────────────────────────────────────────
    yp_visc   = np.linspace(0.5, 5.0, 60)
    yp_log    = np.logspace(np.log10(30), np.log10(2000), 300)
    yp_theory = np.logspace(np.log10(0.5), np.log10(2000), 600)
    up_theory = u_plus_theory(yp_theory)

    # ── 描画 ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))

    # 領域塗り分け
    ax.axvspan(0.5,  5.0, alpha=0.07, color="royalblue",   label="_nolegend_")
    ax.axvspan(5.0, 30.0, alpha=0.07, color="orange",      label="_nolegend_")
    ax.axvspan(30.0, 2000, alpha=0.07, color="limegreen",  label="_nolegend_")

    # 理論線
    lbl_visc = ("粘性底層: $u^+ = y^+$"   if use_jp else "Viscous sublayer: $u^+=y^+$")
    lbl_log  = (r"対数則: $u^+=\frac{1}{\kappa}\ln y^++B$"
                r"  $(\kappa=0.41,\ B=5.0)$" if use_jp else
                r"Log-law: $u^+=\frac{1}{\kappa}\ln y^++B$  $(\kappa=0.41,\ B=5.0)$")
    lbl_cfd  = ("Fluent CFD (k-ω SST, 壁面解像)" if use_jp else
                "Fluent CFD (k-ω SST, wall-resolved)")

    ax.semilogx(yp_visc, yp_visc, "b-",  lw=2.5, label=lbl_visc)
    ax.semilogx(yp_log,  (1.0/KAPPA)*np.log(yp_log)+B_LOG,
                "g-", lw=2.5, label=lbl_log)

    # CFD データ
    ax.semilogx(y_plus_cfd, u_plus_cfd,
                "ro-", ms=4, lw=1.8, label=lbl_cfd, zorder=6)

    # 境界線
    for xv, col in [(5.0, "royalblue"), (30.0, "limegreen")]:
        ax.axvline(xv, color=col, ls=":", lw=1.2, alpha=0.7)

    # 軸設定
    ax.set_xlim(0.5, 2000)
    ax.set_ylim(0, 32)
    ax.set_xlabel("$y^+$ (無次元壁面距離)" if use_jp else "$y^+$ (dimensionless wall distance)",
                  fontsize=13)
    ax.set_ylabel("$u^+$ (無次元流速)" if use_jp else "$u^+$ (dimensionless velocity)",
                  fontsize=13)

    title_main = ("乱流壁法則: 理論解 vs Ansys Fluent CFD  [k-ω SST, 壁面解像]"
                  if use_jp else "Turbulent Wall Function: Theory vs Ansys Fluent CFD  [k-ω SST, wall-resolved]")
    title_sub  = (f"2D チャネル流れ  $Re_D = {Re_D:,}$,  $Re_\\tau = {Re_tau}$"
                  if use_jp else
                  f"2D Channel Flow  $Re_D = {Re_D:,}$,  $Re_\\tau = {Re_tau}$")
    ax.set_title(f"{title_main}\n{title_sub}", fontsize=13)
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(True, which="both", alpha=0.3)

    # ゾーンラベル
    ax.text(1.5, 1.5,  "粘性底層\n$y^+<5$"    if use_jp else "Viscous\nsublayer",
            fontsize=8.5, color="royalblue", ha="center")
    ax.text(12.0, 2.5, "バッファ層"            if use_jp else "Buffer\nlayer",
            fontsize=8.5, color="darkorange", ha="center")
    ax.text(300,  3.5, "対数則領域\n$y^+>30$"  if use_jp else "Log-law\nregion",
            fontsize=8.5, color="darkgreen", ha="center")

    # 数値サマリー
    stats = (f"$\\tau_w = {tau_w:.3f}$ Pa\n"
             f"$u_\\tau = {u_tau:.3f}$ m/s\n"
             f"$Re_\\tau = {Re_tau}$\n"
             f"$k$–$\\varepsilon$ 標準壁関数" if use_jp else
             f"$\\tau_w = {tau_w:.3f}$ Pa\n"
             f"$u_\\tau = {u_tau:.3f}$ m/s\n"
             f"$Re_\\tau = {Re_tau}$\n"
             "k-eps Std. WF")
    ax.text(0.97, 0.04, stats, transform=ax.transAxes,
            fontsize=10, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                      edgecolor="gray", alpha=0.85))

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    print(f"\n[plot] グラフ保存: {OUTPUT_PNG}")
    plt.close()
    return OUTPUT_PNG


# ═══════════════════════════════════════════════════════════════════════════════
# メイン
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="壁法則 理論解 vs Fluent CFD 比較プログラム"
    )
    parser.add_argument("--mesh-only",  action="store_true",
                        help="メッシュ生成のみ実行")
    parser.add_argument("--plot-only",  action="store_true",
                        help="既存データからプロットのみ作成")
    parser.add_argument("--no-fluent",  action="store_true",
                        help="Fluent を実行せず理論解のみプロット")
    args = parser.parse_args()

    sep = "=" * 62

    if args.plot_only:
        print(sep)
        print("プロット生成")
        print(sep)
        plot_comparison()
        return

    if args.no_fluent:
        _plot_theory_only()
        return

    # ── Step 1: メッシュ生成 ───────────────────────────────────────────
    print(sep)
    print("Step 1/3  メッシュ生成")
    print(sep)
    generate_channel_mesh(MESH_FILE, NX, NY, CHANNEL_LENGTH, CHANNEL_HEIGHT)

    if args.mesh_only:
        return

    # ── Step 2: ジャーナル生成 & Fluent 実行 ───────────────────────────
    print(f"\n{sep}")
    print("Step 2/3  Fluent シミュレーション実行")
    print(sep)
    write_fluent_journal(JOURNAL_FILE, MESH_FILE, VELOCITY_FILE, WALL_SHEAR_FILE)
    run_fluent_simulation()


    # ── Step 3: プロット ───────────────────────────────────────────────
    print(f"\n{sep}")
    print("Step 3/3  結果プロット生成")
    print(sep)
    out = plot_comparison()

    print(f"\n{sep}")
    print("完了  出力ファイル一覧:")
    for f in [MESH_FILE, JOURNAL_FILE, VELOCITY_FILE, WALL_SHEAR_FILE, out]:
        exists = "✓" if os.path.exists(f) else "✗"
        print(f"  {exists}  {f}")
    print(sep)


def _plot_theory_only():
    """Fluent なしで理論解のみをプロット (--no-fluent オプション)"""
    font_name = _setup_japanese_font()
    use_jp = font_name is not None

    yp = np.logspace(np.log10(0.5), np.log10(2000), 600)
    up = u_plus_theory(yp)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.axvspan( 0.5,  5.0, alpha=0.08, color="royalblue")
    ax.axvspan( 5.0, 30.0, alpha=0.08, color="orange")
    ax.axvspan(30.0, 2000, alpha=0.08, color="limegreen")

    yp_v = np.linspace(0.5, 5.0, 60)
    yp_l = np.logspace(np.log10(30), np.log10(2000), 300)
    ax.semilogx(yp_v, yp_v,                              "b-", lw=2.5,
                label="粘性底層: $u^+=y^+$" if use_jp else "Viscous sublayer: $u^+=y^+$")
    ax.semilogx(yp_l, (1/KAPPA)*np.log(yp_l)+B_LOG, "g-", lw=2.5,
                label=r"対数則: $u^+=\frac{1}{\kappa}\ln y^++B$" if use_jp else
                      r"Log-law: $u^+=\frac{1}{\kappa}\ln y^++B$")
    ax.semilogx(yp, up, "k--", lw=1.2, alpha=0.6, label="複合壁法則" if use_jp else "Composite")

    ax.set_xlim(0.5, 2000); ax.set_ylim(0, 30)
    ax.set_xlabel("$y^+$", fontsize=13)
    ax.set_ylabel("$u^+$", fontsize=13)
    ax.set_title("壁法則 理論解" if use_jp else "Wall Law (Theory Only)", fontsize=13)
    ax.legend(fontsize=11); ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(WORK_DIR, "wall_function_theory.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[plot] 理論解プロット保存: {out}")
    plt.close()


if __name__ == "__main__":
    main()
