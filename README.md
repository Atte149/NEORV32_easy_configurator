# NEORV32_easy_configurator
Selection of frequency, cache sizes, and memory usage for various FPGAs with NEORV32. This is my academic project

Quick start
# Interactive mode (recommended)
python advisor.py

# CLI: 666-byte binary, external Flash, math profile, GW1NR-9 crystal
python advisor.py --binary 666 --flash --profile math --crystal GW1NR-9

# JSON output for integration with synthesis scripts
python advisor.py --binary 4096 --flash --profile balanced --json

# List of supported crystals
python advisor.py --list-crystals

Dependencies: only the standard Python 3.8+ library. No installation is required.

Input parameters
Parameter CLI-flag Description
Binary size --binary BYTES Size of the compiled ELF/binary (bytes)
Program memory type --flash Program in external Flash (uFlash, SPI) vs SRAM
Load profile --profile {math,memory,balanced} Nature of calculations
Crystal --crystal NAME Target FPGA crystal
Reference clock --clock MHZ Clock to PLL (MHz)
BSRAM limit --max-bsram PCT Maximum BSRAM occupation (%)
Target CM/MHz --target-cm VALUE Minimum CoreMark/MHz (optional)
