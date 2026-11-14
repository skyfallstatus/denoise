# Denoise

A noise reduction effect for Premiere Pro on Apple Silicon Macs. Free and open source.

## Using it

Drop **Denoise** on a clip and play it back. Start there; most shots need nothing more.

When a shot does need a hand, the panel has six controls:

- **Look** is your starting point. *Natural* is the default. *Cleaner* keeps less grain, *Keep texture* cleans brightness more gently, *Phone* also goes after the coarser, blotchy grain phones make, and *Colour only* leaves brightness alone.
- **Luma reduction** is how much grain comes out of the brightness, in dB. Every 6 dB halves what's left. 14 is the default, and 30 asks for everything it can take.
- **Colour reduction** does the same for colour blotches. It starts at 30, because colour noise is rarely worth keeping.
- **Keep grain** leaves a thin layer of the original fine grain, so skin doesn't turn to plastic. Set it to 0 to remove that layer. It never adds grain.
- **View** shows you what's happening. *Removed noise* is the useful one: if you can see faces or edges in it, you're taking too much. The view you leave selected is the one that gets exported, so set it back to *Result*.
- **Help** opens a short guide to every control.

Everything else is folded away in **Advanced**. You shouldn't need it often.

## What to expect

On real night footage (Sony S-Log3, and high-ISO 8-bit H.264), with the default settings:

- fine grain in flat areas drops by 11 to 13 dB, leaving about a quarter of it;
- colour noise drops by 6 to 10 dB, leaving a half to a third;
- frame-to-frame shimmer in still areas is 2 to 3 times lower.

On an M5 Max, 4K at 25 fps plays in real time at full resolution. For 4K at 50 fps, set playback to 1/2. Exporting 5 seconds of 4K 50p takes 3 to 6 seconds.

## How it works

It measures the noise in every frame, by colour channel, grain size and brightness. It lines up the neighbouring frames and averages what matches, which is what stops still areas from shimmering. Whatever is left gets cleaned inside the frame, with that measurement deciding what is grain and what is detail. The full story, with diagrams and equations, is in [docs/ALGORITHM.md](docs/ALGORITHM.md).

## Install

1. Download **Denoise-0.1.0.zip** from [Releases](https://github.com/skyfallstatus/denoise/releases/latest) and open it.
2. Quit Premiere.
3. Open Terminal, type `zsh ` (with a space), drag **install.sh** from the **Denoise-0.1.0** folder into the Terminal window, and press Return.

The script asks for your Mac password, because Adobe's shared plug-in folder belongs to the system. Open Premiere again and you'll find the effect under **Video Effects > Denoise**.

To build it from the source code instead, see [docs/BUILDING.md](docs/BUILDING.md).

## Good to know

- It runs on Apple Silicon Macs in Premiere Pro, and is tested on Premiere Pro 27.
- It isn't signed by Apple yet, which is why it installs from Terminal rather than with an installer.
- 4K 50p doesn't play in real time at full resolution yet.
- In dark, flat shadows it keeps more fine texture than some denoisers. For a smoother look, raise Luma reduction or lower Keep grain.

If a shot comes out wrong, open an issue with a frame from it. That's the most useful thing you can send.

## Licence

MIT. Use it, change it, share it. See [LICENSE](LICENSE).
