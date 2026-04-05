#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NEORV32 Configuration Advisor
==============================
Автоматизированный подбор параметров конфигурации процессорного ядра NEORV32
на основе экспериментальных данных параметрического исследования.

Автор: [ваше имя]
Источник данных: выпускная квалификационная работа «Параметрическое исследование
производительности RISC-V процессорного ядра NEORV32 на FPGA-кристалле Gowin GW1NR-9C»

GitHub: https://github.com/[username]/neorv32-advisor
"""

import argparse
import sys
import json
import math
from dataclasses import dataclass, field, asdict
from typing import Optional

# ══════════════════════════════════════════════════════════════════════════════
#  БАЗА ДАННЫХ КРИСТАЛЛОВ
# ══════════════════════════════════════════════════════════════════════════════

CRYSTALS = {
    "GW1NR-9": {
        "name":        "Gowin GW1NR-9C (Tang Nano 9K)",
        "lut4":        8640,
        "ff":          6480,
        "bsram":       26,          # блоков по 18 Кбит
        "bsram_kb":    18,          # Кбит на блок
        "dsp":         2,
        "has_pll":     True,
        "fmax_mhz":    54.5,        # экспериментально установлено
        "base_lut":    3829,        # занято ядром NEORV32 в базовой конфигурации
        "base_bsram":  22,          # занято ядром в базовой конфигурации
        "base_ff":     2066,
    },
    "GW2A-18": {
        "name":        "Gowin GW2A-18 (Tang Primer 20K)",
        "lut4":        20736,
        "ff":          15552,
        "bsram":       48,
        "bsram_kb":    18,
        "dsp":         48,
        "has_pll":     True,
        "fmax_mhz":    65.0,
        "base_lut":    3829,
        "base_bsram":  22,
        "base_ff":     2066,
    },
    "iCE40UP5K": {
        "name":        "Lattice iCE40UP5K",
        "lut4":        5280,
        "ff":          5280,
        "bsram":       30,          # 4 Кбит-блоков (EBR)
        "bsram_kb":    4,
        "dsp":         8,
        "has_pll":     True,
        "fmax_mhz":    40.0,
        "base_lut":    3829,
        "base_bsram":  19,          # пересчитано под 4-Кбит блоки
        "base_ff":     2066,
    },
    "ECP5-25F": {
        "name":        "Lattice ECP5-25F",
        "lut4":        24288,
        "ff":          24288,
        "bsram":       56,          # блоков 18 Кбит
        "bsram_kb":    18,
        "dsp":         28,
        "has_pll":     True,
        "fmax_mhz":    80.0,
        "base_lut":    3829,
        "base_bsram":  22,
        "base_ff":     2066,
    },
    "custom": {
        "name":        "Пользовательская платформа",
        "lut4":        None,
        "ff":          None,
        "bsram":       None,
        "bsram_kb":    18,
        "dsp":         None,
        "has_pll":     True,
        "fmax_mhz":    50.0,
        "base_lut":    3829,
        "base_bsram":  22,
        "base_ff":     2066,
    },
}

# ══════════════════════════════════════════════════════════════════════════════
#  ПРОГНОСТИЧЕСКАЯ МОДЕЛЬ (из экспериментальных данных)
# ══════════════════════════════════════════════════════════════════════════════

# Базовые параметры (конфигурация A — без кешей, без DSP, 27 МГц)
BASE_CPI            = 6.58
BASE_CM_MHZ         = 0.511
DELTA_CPI_DSP       = -1.84   # CPU_FAST_MUL_EN=true → CPI снижается на 1.84
DELTA_CPI_ICACHE_OK = -1.12   # ICACHE достаточного размера → снижение CPI
DELTA_CPI_ICACHE_THRASH = -0.07  # ICACHE слишком мал (thrashing)
DELTA_CPI_DCACHE_RAM16  = +3.13  # DCACHE на RAM16 → CPI растёт

# Коэффициент насыщения кеша: кеш должен быть в K раз больше бинарника
CACHE_SATURATION_RATIO = 6.0   # из опыта F5: 4096/666 = 6.1×

# Ресурсы кеша (данные из таблиц 5.2, 5.4)
ICACHE_LUT_OVERHEAD = 312       # LUT на логику управления кешем инструкций
DCACHE_LUT_OVERHEAD = 642       # LUT на кеш данных (включая RAM16)

def icache_blocks_needed(size_bytes: int, crystal: dict) -> int:
    """Минимальное число блоков BSRAM для кеша инструкций нужного размера."""
    target_bytes = size_bytes * CACHE_SATURATION_RATIO
    bsram_bytes  = crystal["bsram_kb"] * 1024 / 8
    return max(1, math.ceil(target_bytes / bsram_bytes))

def icache_size_bytes(num_blocks: int, crystal: dict) -> int:
    return int(num_blocks * crystal["bsram_kb"] * 1024 / 8)

def nearest_power2_blocks(n: int) -> int:
    """Округлить число блоков до степени двойки (ограничения NEORV32)."""
    if n <= 1: return 1
    p = 1
    while p < n:
        p <<= 1
    return p

def predict_cpi(icache_effective: bool, dcache_on_ram16: bool, dsp_enabled: bool) -> float:
    cpi = BASE_CPI
    if icache_effective:
        cpi += DELTA_CPI_ICACHE_OK
    else:
        cpi += DELTA_CPI_ICACHE_THRASH
    if dsp_enabled:
        cpi += DELTA_CPI_DSP
    if dcache_on_ram16:
        cpi += DELTA_CPI_DCACHE_RAM16
    return max(1.0, cpi)

def predict_cm_mhz(cpi: float) -> float:
    """CoreMark/MHz по формуле из работы: CM/MHz ≈ 0.511 × (6.58 / CPI)."""
    return BASE_CM_MHZ * (BASE_CPI / cpi)

# ══════════════════════════════════════════════════════════════════════════════
#  ВХОДНЫЕ ДАННЫЕ
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class UserInput:
    # Приложение
    binary_size_bytes:    int            # размер ELF/бинарника программы
    external_flash:       bool           # программа в uFlash/SPI Flash, не в SRAM
    workload_profile:     str            # "math" | "memory" | "balanced"

    # Платформа
    crystal_key:          str            # ключ из CRYSTALS
    crystal_custom:       Optional[dict] # если crystal_key == "custom"
    clock_mhz:            float          # опорная частота (до PLL)

    # Ограничения
    max_bsram_pct:        float = 95.0   # максимум занятости BSRAM (%)
    target_cm_mhz:        Optional[float] = None  # целевой CM/MHz (опционально)
    max_power_w:          Optional[float] = None  # ограничение по мощности (опционально)

    def crystal(self) -> dict:
        if self.crystal_key == "custom":
            return self.crystal_custom
        return CRYSTALS[self.crystal_key]

# ══════════════════════════════════════════════════════════════════════════════
#  РЕЗУЛЬТАТ
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Recommendation:
    # VHDL-параметры
    mem_int_imem_en:     bool
    icache_en:           bool
    icache_num_blocks:   int
    icache_block_size:   int
    dcache_en:           bool
    cpu_fast_mul_en:     bool
    clock_frequency_hz:  int

    # Предсказание
    predicted_cpi:       float
    predicted_cm_mhz:    float
    predicted_speedup:   float   # относительно базовой A

    # Ресурсы
    lut4_used:           int
    lut4_total:          int
    bsram_used:          int
    bsram_total:         int
    bsram_pct:           float
    dsp_used:            int

    # Предупреждения
    warnings:            list = field(default_factory=list)
    notes:               list = field(default_factory=list)

# ══════════════════════════════════════════════════════════════════════════════
#  ЯДРО АЛГОРИТМА
# ══════════════════════════════════════════════════════════════════════════════

def compute(inp: UserInput) -> Recommendation:
    cr       = inp.crystal()
    warnings = []
    notes    = []

    total_bsram = cr["bsram"]
    base_bsram  = cr["base_bsram"]
    base_lut    = cr["base_lut"]
    avail_bsram = total_bsram - base_bsram  # свободно после ядра

    max_bsram_absolute = int(total_bsram * inp.max_bsram_pct / 100)
    budget_bsram       = max_bsram_absolute - base_bsram   # бюджет для кешей

    # ── 1. MEM_INT_IMEM_EN ─────────────────────────────────────────────────
    mem_int_imem_en = not inp.external_flash
    if not inp.external_flash:
        notes.append(
            "Программа в SRAM (MEM_INT_IMEM_EN=true): кеш инструкций не даёт "
            "прироста — все обращения уже однотактовые."
        )

    # ── 2. ICACHE ──────────────────────────────────────────────────────────
    icache_en         = False
    icache_num_blocks = 0
    icache_block_size = 64
    icache_effective  = False

    if inp.external_flash:
        needed_blocks = icache_blocks_needed(inp.binary_size_bytes, cr)
        needed_blocks = nearest_power2_blocks(needed_blocks)

        if needed_blocks > budget_bsram:
            # Урезаем до бюджета
            actual_blocks = nearest_power2_blocks(budget_bsram)
            if actual_blocks < 1:
                actual_blocks = 0
                warnings.append(
                    "Недостаточно свободных блоков BSRAM для кеша инструкций. "
                    "ICACHE_EN = false."
                )
            else:
                icache_en         = True
                icache_num_blocks = actual_blocks
                actual_size       = icache_size_bytes(actual_blocks, cr)
                ratio             = actual_size / inp.binary_size_bytes
                icache_effective  = ratio >= CACHE_SATURATION_RATIO
                if not icache_effective:
                    warnings.append(
                        f"Кеш ({actual_size} байт) меньше рекомендуемых "
                        f"{CACHE_SATURATION_RATIO:.0f}× бинарника "
                        f"({inp.binary_size_bytes * CACHE_SATURATION_RATIO:.0f} байт). "
                        f"Возможен режим thrashing, прирост минимален."
                    )
        else:
            icache_en         = True
            icache_num_blocks = needed_blocks
            icache_effective  = True

        if icache_en:
            # Размер строки: 128 байт если кеш ≥ 4 КБ (из опытов F5/F6)
            icache_size = icache_size_bytes(icache_num_blocks, cr)
            icache_block_size = 128 if icache_size >= 4096 else 64
            notes.append(
                f"Кеш инструкций: {icache_num_blocks} блоков × "
                f"{icache_block_size} байт = {icache_size} байт "
                f"({icache_size / inp.binary_size_bytes:.1f}× бинарника)."
            )

    bsram_after_icache = base_bsram + (icache_num_blocks if icache_en else 0)

    # ── 3. DCACHE ──────────────────────────────────────────────────────────
    dcache_en       = False
    dcache_on_ram16 = False
    remaining       = max_bsram_absolute - bsram_after_icache

    if inp.workload_profile == "memory" and remaining >= 4:
        dcache_en = True
        notes.append(
            f"Профиль 'memory': DCACHE_EN = true. "
            f"Доступно {remaining} блоков BSRAM."
        )
    elif inp.workload_profile == "memory" and remaining < 4:
        dcache_en       = True
        dcache_on_ram16 = True
        warnings.append(
            "КРИТИЧНО: DCACHE_EN=true при недостатке BSRAM приведёт к реализации "
            "кеша данных на RAM16 (LUT-ткань). По данным эксперимента C это "
            "СНИЖАЕТ производительность на ~32% (CPI растёт с 6.58 до 9.71). "
            "Настоятельно рекомендуется DCACHE_EN = false."
        )
        dcache_en = False   # автоматически выключаем — это безопаснее
    else:
        notes.append(
            "DCACHE_EN = false: профиль нагрузки не требует кеша данных "
            "или недостаточно ресурсов BSRAM."
        )

    # ── 4. DSP ─────────────────────────────────────────────────────────────
    dsp_available  = cr.get("dsp", 0) or 0
    cpu_fast_mul   = False
    dsp_used       = 0

    if inp.workload_profile in ("math", "balanced") and dsp_available >= 1:
        cpu_fast_mul = True
        dsp_used     = 1
        notes.append(
            "CPU_FAST_MUL_EN = true: DSP-блок переносит умножение на "
            "аппаратный умножитель, освобождает ~136 LUT, снижает CPI на 1.84."
        )
    elif dsp_available == 0:
        notes.append(
            "CPU_FAST_MUL_EN = false: DSP-блоки отсутствуют на данной платформе."
        )
    else:
        notes.append(
            "CPU_FAST_MUL_EN = false: профиль 'memory' — умножение не доминирует."
        )

    # ── 5. Тактовая частота ────────────────────────────────────────────────
    fmax    = cr.get("fmax_mhz", 50.0)
    has_pll = cr.get("has_pll", False)

    if has_pll:
        clock_hz = int(fmax * 1_000_000)
        notes.append(
            f"Рекомендуется максимальная частота через PLL: {fmax} МГц."
        )
    else:
        clock_hz = int(inp.clock_mhz * 1_000_000)
        notes.append(
            f"PLL недоступен, используется опорная частота: {inp.clock_mhz} МГц."
        )

    # ── 6. Предсказание производительности ─────────────────────────────────
    icache_flag = icache_effective if icache_en else False
    cpi_pred    = predict_cpi(icache_flag, dcache_on_ram16, cpu_fast_mul)
    cm_mhz_pred = predict_cm_mhz(cpi_pred)
    speedup     = cm_mhz_pred / BASE_CM_MHZ

    if inp.target_cm_mhz and cm_mhz_pred < inp.target_cm_mhz:
        warnings.append(
            f"Целевой CM/MHz = {inp.target_cm_mhz:.3f} не достигается. "
            f"Предсказано: {cm_mhz_pred:.3f}. "
            "Рассмотрите: повышение частоты (если есть запас Fmax), "
            "увеличение кеша, активацию DSP."
        )

    # ── 7. Ресурсы ─────────────────────────────────────────────────────────
    lut_delta  = (-136 if cpu_fast_mul else 0)
    lut_delta += (ICACHE_LUT_OVERHEAD if icache_en else 0)

    lut_used   = base_lut + lut_delta
    bsram_used = bsram_after_icache
    bsram_pct  = bsram_used / total_bsram * 100

    lut_total  = cr.get("lut4") or 9999

    if lut_used / lut_total > 0.80:
        warnings.append(
            f"Высокая загрузка LUT: {lut_used}/{lut_total} "
            f"({lut_used/lut_total*100:.0f}%). Могут возникнуть проблемы с P&R."
        )

    return Recommendation(
        mem_int_imem_en    = mem_int_imem_en,
        icache_en          = icache_en,
        icache_num_blocks  = icache_num_blocks if icache_en else 0,
        icache_block_size  = icache_block_size,
        dcache_en          = dcache_en,
        cpu_fast_mul_en    = cpu_fast_mul,
        clock_frequency_hz = clock_hz,
        predicted_cpi      = round(cpi_pred, 2),
        predicted_cm_mhz   = round(cm_mhz_pred, 3),
        predicted_speedup  = round(speedup, 2),
        lut4_used          = lut_used,
        lut4_total         = lut_total,
        bsram_used         = bsram_used,
        bsram_total        = total_bsram,
        bsram_pct          = round(bsram_pct, 1),
        dsp_used           = dsp_used,
        warnings           = warnings,
        notes              = notes,
    )

# ══════════════════════════════════════════════════════════════════════════════
#  ФОРМАТИРОВАНИЕ ВЫВОДА
# ══════════════════════════════════════════════════════════════════════════════

BOLD   = "\033[1m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
RED    = "\033[31m"
CYAN   = "\033[36m"
RESET  = "\033[0m"
DIM    = "\033[2m"

def bool_str(v: bool) -> str:
    return f"{GREEN}true{RESET}" if v else f"false"

def pct_bar(pct: float, width: int = 20) -> str:
    filled = int(pct / 100 * width)
    color  = RED if pct > 90 else (YELLOW if pct > 75 else GREEN)
    return f"{color}{'█' * filled}{'░' * (width - filled)}{RESET} {pct:.1f}%"

def print_report(inp: UserInput, rec: Recommendation, crystal: dict):
    W = 65
    line = "─" * W

    print(f"\n{BOLD}{'═' * W}{RESET}")
    print(f"{BOLD}  NEORV32 Configuration Advisor{RESET}")
    print(f"{DIM}  Кристалл: {crystal['name']}{RESET}")
    print(f"{'═' * W}{RESET}")

    # ── Входные данные ──────────────────────────────────────────────────────
    print(f"\n{BOLD}{CYAN}  ВХОДНЫЕ ДАННЫЕ{RESET}")
    print(f"  {line}")
    print(f"  Размер бинарника       {BOLD}{inp.binary_size_bytes:>8} байт{RESET}  "
          f"({inp.binary_size_bytes/1024:.2f} КБ)")
    print(f"  Память программ        {'внешняя (uFlash/SPI)' if inp.external_flash else 'внутренняя SRAM'}")
    print(f"  Профиль нагрузки       {inp.workload_profile}")
    print(f"  Опорная частота        {inp.clock_mhz} МГц")

    # ── Рекомендованные параметры ───────────────────────────────────────────
    print(f"\n{BOLD}{CYAN}  РЕКОМЕНДОВАННЫЕ ПАРАМЕТРЫ NEORV32{RESET}")
    print(f"  {line}")
    params = [
        ("MEM_INT_IMEM_EN",   bool_str(rec.mem_int_imem_en)),
        ("ICACHE_EN",         bool_str(rec.icache_en)),
        ("ICACHE_NUM_BLOCKS", f"{BOLD}{rec.icache_num_blocks}{RESET}" if rec.icache_en else f"{DIM}—{RESET}"),
        ("ICACHE_BLOCK_SIZE", f"{BOLD}{rec.icache_block_size}{RESET}" if rec.icache_en else f"{DIM}—{RESET}"),
        ("DCACHE_EN",         bool_str(rec.dcache_en)),
        ("CPU_FAST_MUL_EN",   bool_str(rec.cpu_fast_mul_en)),
        ("CLOCK_FREQUENCY",   f"{BOLD}{rec.clock_frequency_hz:_}{RESET} Гц  "
                               f"({rec.clock_frequency_hz/1e6:.1f} МГц)"),
    ]
    for name, val in params:
        print(f"  {name:<25} {val}")

    # ── VHDL сниппет ────────────────────────────────────────────────────────
    print(f"\n{BOLD}{CYAN}  ГОТОВЫЙ VHDL СНИППЕТ{RESET}")
    print(f"  {line}")
    vhdl = f"""\
  -- Вставьте в инстанциирование neorv32_top:
  MEM_INT_IMEM_EN    => {'true' if rec.mem_int_imem_en else 'false'},
  ICACHE_EN          => {'true' if rec.icache_en else 'false'},
  ICACHE_NUM_BLOCKS  => {rec.icache_num_blocks if rec.icache_en else 0},
  ICACHE_BLOCK_SIZE  => {rec.icache_block_size},
  DCACHE_EN          => {'true' if rec.dcache_en else 'false'},
  CPU_FAST_MUL_EN    => {'true' if rec.cpu_fast_mul_en else 'false'},
  CLOCK_FREQUENCY    => {rec.clock_frequency_hz},"""
    for l in vhdl.split("\n"):
        print(f"  {DIM}{l}{RESET}")

    # ── Предсказание ─────────────────────────────────────────────────────────
    print(f"\n{BOLD}{CYAN}  ПРОГНОЗ ПРОИЗВОДИТЕЛЬНОСТИ{RESET}")
    print(f"  {line}")
    speedup_color = GREEN if rec.predicted_speedup >= 1.2 else (YELLOW if rec.predicted_speedup >= 1.0 else RED)
    print(f"  CoreMark/MHz           {BOLD}{rec.predicted_cm_mhz:.3f}{RESET}")
    print(f"  CPI (такт/инструкцию)  {BOLD}{rec.predicted_cpi:.2f}{RESET}")
    print(f"  Ускорение vs базовая   {speedup_color}{BOLD}×{rec.predicted_speedup:.2f}{RESET}")
    print(f"  {DIM}Базовая конфигурация A: CM/MHz=0.511, CPI=6.58{RESET}")
    print(f"  {DIM}Модель: CM/MHz = 0.511 × (6.58 / CPI), погрешность ±5%{RESET}")

    # ── Ресурсы ──────────────────────────────────────────────────────────────
    print(f"\n{BOLD}{CYAN}  АППАРАТНЫЕ РЕСУРСЫ{RESET}")
    print(f"  {line}")
    lut_pct = rec.lut4_used / rec.lut4_total * 100
    print(f"  LUT4     {rec.lut4_used:>5}/{rec.lut4_total:<6}  {pct_bar(lut_pct)}")
    print(f"  BSRAM    {rec.bsram_used:>5}/{rec.bsram_total:<6}  {pct_bar(rec.bsram_pct)}")
    print(f"  DSP      {rec.dsp_used:>5}/{crystal.get('dsp',0):<6}")

    # ── Примечания ───────────────────────────────────────────────────────────
    if rec.notes:
        print(f"\n{BOLD}{CYAN}  ПРИМЕЧАНИЯ{RESET}")
        print(f"  {line}")
        for note in rec.notes:
            for i, part in enumerate(_wrap(note, W - 6)):
                prefix = "  ○ " if i == 0 else "    "
                print(f"{prefix}{part}")

    # ── Предупреждения ───────────────────────────────────────────────────────
    if rec.warnings:
        print(f"\n{BOLD}{RED}  ⚠  ПРЕДУПРЕЖДЕНИЯ{RESET}")
        print(f"  {line}")
        for w in rec.warnings:
            for i, part in enumerate(_wrap(w, W - 6)):
                prefix = f"  {YELLOW}▲{RESET} " if i == 0 else "    "
                print(f"{prefix}{part}")

    print(f"\n{'═' * W}\n")


def _wrap(text: str, width: int) -> list:
    """Простой перенос строк."""
    words  = text.split()
    lines  = []
    cur    = ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            if cur:
                lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return lines or [""]

# ══════════════════════════════════════════════════════════════════════════════
#  ИНТЕРАКТИВНЫЙ РЕЖИМ
# ══════════════════════════════════════════════════════════════════════════════

def ask(prompt: str, default=None, cast=str, choices=None):
    """Задать вопрос пользователю."""
    if default is not None:
        prompt_full = f"  {prompt} [{default}]: "
    else:
        prompt_full = f"  {prompt}: "
    while True:
        try:
            raw = input(prompt_full).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nПрервано.")
            sys.exit(0)
        if raw == "" and default is not None:
            return cast(default)
        if raw == "" and default is None:
            print("  Значение обязательно.")
            continue
        if choices and raw not in choices:
            print(f"  Допустимые значения: {', '.join(choices)}")
            continue
        try:
            return cast(raw)
        except (ValueError, TypeError):
            print(f"  Неверный формат. Ожидается {cast.__name__}.")


def interactive_mode() -> UserInput:
    print(f"\n{BOLD}  NEORV32 Configuration Advisor — интерактивный режим{RESET}\n")

    # Кристалл
    crystal_choices = list(CRYSTALS.keys())
    print(f"  Доступные кристаллы:")
    for k, v in CRYSTALS.items():
        print(f"    {BOLD}{k:<12}{RESET}  {v['name']}")
    crystal_key = ask("Выберите кристалл", default="GW1NR-9", choices=crystal_choices)

    crystal_custom = None
    if crystal_key == "custom":
        print("\n  Введите параметры платформы:")
        cr = dict(CRYSTALS["custom"])
        cr["lut4"]      = ask("  LUT4 (всего)", cast=int)
        cr["ff"]        = ask("  Flip-Flop (всего)", cast=int)
        cr["bsram"]     = ask("  Блоков BSRAM", cast=int)
        cr["bsram_kb"]  = ask("  Размер блока BSRAM (Кбит)", default=18, cast=int)
        cr["dsp"]       = ask("  DSP-блоков", default=0, cast=int)
        cr["has_pll"]   = ask("  Есть PLL?", default="да", choices=["да","нет"]) == "да"
        cr["fmax_mhz"]  = ask("  Fmax (МГц)", default=50.0, cast=float)
        cr["base_lut"]  = ask("  LUT занято базовым ядром NEORV32", default=3829, cast=int)
        cr["base_bsram"]= ask("  BSRAM занято базовым ядром", default=22, cast=int)
        cr["name"]      = "Пользовательская платформа"
        crystal_custom  = cr

    print()
    binary_size = ask("Размер бинарника программы (байт)", cast=int)

    ext_flash_raw = ask(
        "Программа хранится во внешней/встроенной Flash? (да=uFlash/SPI, нет=SRAM)",
        default="да", choices=["да","нет"]
    )
    external_flash = (ext_flash_raw == "да")

    print(f"""
  Профили нагрузки:
    math     — умножение/сдвиги доминируют (матрицы, DSP, криптография)
    memory   — интенсивный доступ к данным (сортировки, буферы, linked lists)
    balanced — смешанная нагрузка (CoreMark, типичные приложения)""")
    workload = ask("Профиль нагрузки", default="balanced",
                   choices=["math","memory","balanced"])

    cr = crystal_custom or CRYSTALS[crystal_key]
    clock_mhz = ask(
        f"Опорная тактовая частота (МГц)",
        default=27.0, cast=float
    )

    max_bsram = ask("Максимальная занятость BSRAM (%)", default=95.0, cast=float)

    target_raw = ask(
        "Целевой CoreMark/MHz (Enter — пропустить)",
        default="", cast=str
    )
    target_cm = float(target_raw) if target_raw else None

    return UserInput(
        binary_size_bytes  = binary_size,
        external_flash     = external_flash,
        workload_profile   = workload,
        crystal_key        = crystal_key,
        crystal_custom     = crystal_custom,
        clock_mhz          = clock_mhz,
        max_bsram_pct      = max_bsram,
        target_cm_mhz      = target_cm,
    )

# ══════════════════════════════════════════════════════════════════════════════
#  РЕЖИМ СРАВНЕНИЯ
# ══════════════════════════════════════════════════════════════════════════════

def compare_mode(inp: UserInput):
    """Показать таблицу предсказаний для нескольких вариантов."""
    cr = inp.crystal()
    print(f"\n{BOLD}  СРАВНИТЕЛЬНАЯ ТАБЛИЦА КОНФИГУРАЦИЙ{RESET}")
    print(f"  Кристалл: {cr['name']}  |  Бинарник: {inp.binary_size_bytes} байт\n")

    header = f"  {'Конфигурация':<28} {'CM/MHz':>8} {'CPI':>6} {'BSRAM%':>8} {'LUT':>6}  Ускор."
    print(header)
    print("  " + "─" * 65)

    rec = compute(inp)

    def _override(**kwargs):
        base = {k: getattr(inp, k) for k in UserInput.__dataclass_fields__}
        base.update(kwargs)
        return base

    configs = [
        ("A: базовая (без оптимизаций)",
         _override(workload_profile="balanced", clock_mhz=27.0,
                   target_cm_mhz=None, max_power_w=None)),
        ("E: только DSP + barrel shifter",
         _override(workload_profile="math", clock_mhz=27.0)),
        ("★ Рекомендовано", None),
    ]

    for label, override in configs:
        if override is None:
            r = rec
        else:
            try:
                ui = UserInput(**override)
                r  = compute(ui)
            except Exception:
                continue
        bar = "█" * int(r.predicted_speedup * 5)
        print(f"  {label:<28} {r.predicted_cm_mhz:>8.3f} "
              f"{r.predicted_cpi:>6.2f} {r.bsram_pct:>7.1f}% "
              f"{r.lut4_used:>6}  ×{r.predicted_speedup:.2f} {bar}")

# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_parser():
    p = argparse.ArgumentParser(
        prog        = "advisor",
        description = "NEORV32 Configuration Advisor — подбор параметров ядра",
        epilog      = "Без аргументов запускается интерактивный режим.",
        formatter_class = argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--binary",    type=int,   metavar="BYTES",
                   help="Размер бинарника программы (байт)")
    p.add_argument("--flash",     action="store_true",
                   help="Программа в внешней Flash (uFlash/SPI)")
    p.add_argument("--profile",   choices=["math","memory","balanced"],
                   default="balanced", help="Профиль нагрузки (default: balanced)")
    p.add_argument("--crystal",   choices=list(CRYSTALS.keys()),
                   default="GW1NR-9", help="Кристалл FPGA")
    p.add_argument("--clock",     type=float, default=27.0, metavar="MHZ",
                   help="Опорная тактовая частота МГц (default: 27)")
    p.add_argument("--max-bsram", type=float, default=95.0, metavar="PCT",
                   help="Максимум занятости BSRAM %% (default: 95)")
    p.add_argument("--target-cm", type=float, metavar="CM_MHZ",
                   help="Целевой CoreMark/MHz")
    p.add_argument("--json",      action="store_true",
                   help="Вывод в JSON (для скриптов)")
    p.add_argument("--compare",   action="store_true",
                   help="Показать сравнительную таблицу конфигураций")
    p.add_argument("--list-crystals", action="store_true",
                   help="Показать список поддерживаемых кристаллов")
    return p


def main():
    parser = build_parser()
    args   = parser.parse_args()

    if args.list_crystals:
        print(f"\n{'Ключ':<15} {'Кристалл':<35} {'LUT':>7} {'BSRAM':>6} {'DSP':>4} {'Fmax':>6}")
        print("─" * 75)
        for k, v in CRYSTALS.items():
            print(f"{k:<15} {v['name']:<35} "
                  f"{str(v['lut4'] or '?'):>7} "
                  f"{str(v['bsram']) + 'б':>6} "
                  f"{str(v['dsp'] or '?'):>4} "
                  f"{v['fmax_mhz']:>5}МГц")
        print()
        return

    # Если не передан --binary — запускаем интерактивный режим
    if args.binary is None:
        inp = interactive_mode()
    else:
        inp = UserInput(
            binary_size_bytes = args.binary,
            external_flash    = args.flash,
            workload_profile  = args.profile,
            crystal_key       = args.crystal,
            crystal_custom    = None,
            clock_mhz         = args.clock,
            max_bsram_pct     = args.max_bsram,
            target_cm_mhz     = args.target_cm,
        )

    rec    = compute(inp)
    crystal = inp.crystal()

    if args.json:
        out = {
            "input":          asdict(inp),
            "recommendation": asdict(rec),
        }
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    print_report(inp, rec, crystal)

    if args.compare or (args.binary is None):
        compare_mode(inp)


if __name__ == "__main__":
    main()
