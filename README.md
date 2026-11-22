# kscrollshot

Scrolling screenshots for KDE Plasma on Wayland: select an area, scroll it, get one tall PNG.

I asked Claude to build me a scrolling screen capture tool and this is what it produced. Figured others may also find it useful.

Only tested on my PC (CachyOS, KDE Plasma, Wayland) so YMMV. Provided as is.

## Install (Arch / CachyOS)

    sudo pacman -S --needed python-numpy python-gobject python-pyqt6 gst-plugin-pipewire gst-plugins-base
    python kscrollshot.py --install

Optional: `wl-clipboard`, so the result is copied to the clipboard automatically.

## Use

- `kscrollshot`: drag a box over the scrolling area, then scroll down. It finishes when you stop scrolling for 4 seconds, click its tray icon, or run it again.
- `kscrollshot --auto`: scrolls for you. KDE asks for permission to send input.
- Images are saved to `Pictures/Screenshots`. See `kscrollshot --help` for options.

## Known limits

- Vertical scrolling only.
- Loses track if you scroll more than about a screenful at once; scroll back up a little to recover.
- Floating elements (chat bubbles, "back to top" buttons) can leave stripes.

## Licence

MIT
