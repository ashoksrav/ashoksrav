import clr
import time
import math

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DLL_PATH = (
    r"C:\Program Files (x86)\Analog Devices"
    r"\VisualAnalog\HadBoardApi.dll"
)

FPGA_FILE = (
    r"C:\Program Files (x86)\Analog Devices"
    r"\VisualAnalog\Hardware\HSC_ADC_EVALC"
    r"\ad9670_lvds_mar2012.bin"
)

SAMPLE_COUNT = 8192
FILL_TIMEOUT_MS = 1000
FILL_DELAY_MS = 60

# AD9670 configuration
ADC_BITS = 14
ADC_MIDPOINT = 1 << (ADC_BITS - 1)
ADC_MASK = (1 << ADC_BITS) - 1

# --------------------------------------------------
# LOAD DLL AND DETECT BOARD
# --------------------------------------------------

clr.AddReference(DLL_PATH)

from HadBoardApi import HadBoard

HadBoard.FindBoards()

print("Boards found:", HadBoard.BoardCount)

if HadBoard.BoardCount < 1:
    raise RuntimeError("HSC-ADC-EVALC not detected")

HadBoard.SelectBoard(0)
board = HadBoard.Board

# --------------------------------------------------
# FPGA CONFIGURATION
# --------------------------------------------------

print("\nConfiguring FPGA...")

board.ConfigureFpga(FPGA_FILE)
board.SelectOutputWidth(16)

print("FPGA configuration calls completed.")

# --------------------------------------------------
# FIFO FILL
# --------------------------------------------------

print("\nFilling FIFOs...")

board.FillFifos(SAMPLE_COUNT, FILL_TIMEOUT_MS)

time.sleep(FILL_DELAY_MS / 1000.0)

# --------------------------------------------------
# RAW FIFO ANALYSIS
# --------------------------------------------------

def analyze_raw_fifo(words):

    values = [
        int(words[i]) & 0xFFFF
        for i in range(len(words))
    ]

    signed16 = [
        value - 65536 if value >= 32768 else value
        for value in values
    ]

    rms16 = math.sqrt(
        sum(value * value for value in signed16) / len(signed16)
    )

    duplicate_pairs = sum(
        1 for i in range(1, len(values))
        if values[i] == values[i - 1]
    )

    return {
        "nonzero": sum(1 for value in values if value != 0),
        "unique": len(set(values)),
        "minimum": min(signed16),
        "maximum": max(signed16),
        "mean": sum(signed16) / len(signed16),
        "rms": rms16,
        "duplicates": duplicate_pairs,
        "low2": sorted(set(value & 0x03 for value in values)),
        "low4": sorted(set(value & 0x0F for value in values))
    }


# --------------------------------------------------
# AD9670 14-BIT OFFSET-BINARY ANALYSIS
# Candidate alignment: bits 15:2
# --------------------------------------------------

def analyze_ad9670_fifo(words, fifo_number):

    raw = [
        int(words[i]) & 0xFFFF
        for i in range(len(words))
    ]

    # Candidate: 12-bit data aligned to bits 15:4
    code12 = [
        (value >> 4) & 0x0FFF
        for value in raw
    ]

    # Offset-binary centering
    signal = [
        value - 2048
        for value in code12
    ]

    rms = math.sqrt(
        sum(value * value for value in signal) / len(signal)
    )

    print("\n12-BIT OFFSET-BINARY DIAGNOSTIC")
    print("FIFO:", fifo_number)
    print("First 16 codes:", code12[:16])
    print("First 16 centred:", signal[:16])

    print("Unique codes:", len(set(code12)))
    print("Minimum code:", min(code12))
    print("Maximum code:", max(code12))
    print("Mean code:", round(sum(code12) / len(code12), 2))
    print("Centred mean:", round(sum(signal) / len(signal), 2))
    print("Centred RMS:", round(rms, 2))


def inspect_fifo_bits(words, fifo_number):

    print(f"\n========== FIFO {fifo_number} BIT INSPECTION ==========")

    for i in range(64):

        value = int(words[i]) & 0xFFFF

        print(
            f"{i:03d} | "
            f"HEX: 0x{value:04X} | "
            f"BIN: {value:016b} | "
            f"LOW4: {value & 0x0F}"
        )

# --------------------------------------------------
# CAPTURE ALL 8 FIFOs
# --------------------------------------------------

print("\nCapturing all 8 FIFOs...")

captures = {}

for fifo in range(8):

    words = board.CaptureFifoWords(SAMPLE_COUNT, fifo)

    if len(words) != SAMPLE_COUNT:
        raise RuntimeError(
            f"FIFO {fifo}: Expected {SAMPLE_COUNT}, got {len(words)}"
        )

    captures[fifo] = words

# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

print("\n========== FIFO RESULTS ==========")

for fifo in range(8):

    words = captures[fifo]
    stats = analyze_raw_fifo(words)

    first_values = [
        int(words[i]) & 0xFFFF
        for i in range(min(16, len(words)))
    ]

    print("\n================================")
    print("FIFO:", fifo)
    print("Count:", len(words))
    print("Nonzero:", stats["nonzero"])
    print("Unique:", stats["unique"])
    print("First 16:", first_values)

    print("Raw signed minimum:", stats["minimum"])
    print("Raw signed maximum:", stats["maximum"])
    print("Raw signed mean:", round(stats["mean"], 2))
    print("Raw signed RMS:", round(stats["rms"], 2))
    print("Duplicate pairs:", stats["duplicates"])

    print("Lower 2-bit unique:", stats["low2"])
    print("Lower 4-bit unique:", stats["low4"])

    # Analyse the four FIFOs that previously contained varying data.
    if fifo in [2, 3, 6, 7]:
        analyze_ad9670_fifo(words, fifo)
        inspect_fifo_bits(captures[fifo], fifo)

print("\n========== CAPTURE COMPLETE ==========")
