# neatRace

`neatRace` is a small Python experiment that trains a neural network with
NEAT to drive a car around a 2D race track rendered with pygame.

The project uses:

- `pygame` for rendering and collision/radar simulation.
- `neat-python` for evolving the driving neural networks.
- `matplotlib` only when `--plot` is enabled.

## Project Layout

```text
.
├── assets/
│   ├── Car_3_01.png   # Car sprite
│   ├── Track1.png     # Visible track
│   └── fondo1.png     # Road mask: white pixels are valid driving area
├── config.txt         # neat-python configuration
├── neatRace.py        # Training entry point
└── requirements.txt   # Python dependencies
```

## Requirements

- Python 3.10 or newer
- Windows, macOS, or Linux with a working graphical environment for normal play

## Setup

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Start a normal training run:

```bash
python neatRace.py
```

Run a short smoke test without opening a pygame window:

```bash
python neatRace.py --headless --generations 1 --frame-limit 60 --no-radars
```

Run fewer generations while tuning:

```bash
python neatRace.py --generations 5 --frame-limit 400
```

Save a fitness plot after training:

```bash
python neatRace.py --generations 10 --plot --plot-file training-progress.png
```

## How It Works

Each car has three distance sensors:

- 45 degrees left
- straight ahead
- 45 degrees right

The neural network receives those three distances and produces two outputs:

- turn right
- turn left

Cars gain fitness by surviving longer and by reaching checkpoint rectangles in
order around the track. The white path in `assets/fondo1.png` is used as the
road mask, while `assets/Track1.png` is only the visual track.

## Useful CLI Options

```text
--generations N   Number of NEAT generations to train. Default: 50
--frame-limit N   Maximum frames per generation. Default: 800
--headless        Use pygame's dummy video driver for CI/smoke tests
--no-radars       Hide radar debug lines
--seed N          Use a deterministic random seed
--plot            Plot max fitness after training
--plot-file PATH  Save the plot instead of opening it interactively
```

## Notes

The default NEAT population size is intentionally small (`pop_size = 5`) so the
demo starts quickly. Increase `pop_size`, `--generations`, and `--frame-limit`
in `config.txt`/CLI options if you want stronger training runs.
