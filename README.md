# kscrollshot

Scrolling screenshots for KDE Plasma on Wayland: select an area, scroll it, get one tall PNG.

I asked Claude to build me a scrolling screen capture tool and this is what it produced. Figured others may also find it useful.

Only tested on my PC (CachyOS, KDE Plasma, Wayland) so YMMV. Provided as is.

## Install (Arch / CachyOS)

    sudo pacman -S --needed python-numpy python-gobject python-pyqt6 gst-plugin-pipewire gst-plugins-base
    python kscrollshot.py --install

`--install` copies it to `~/.local/bin` and adds a launcher entry. Bind a key under System Settings > Keyboard > Shortcuts; pressing it again mid-capture finishes and saves.

Optional package: `wl-clipboard`, so the result is copied to the clipboard automatically.

`wl-clipboard` is a small package that provides the `wl-copy` command, and kscrollshot uses that command for copying:

- With it installed: when a capture finishes, the image is copied to the clipboard automatically, and it stays there after you close the result window.
- Without it: nothing is copied automatically. The Copy button in the result window still works, but it uses Qt's own clipboard. On Wayland, an app's clipboard contents normally vanish when the app quits, and kscrollshot quits when you close that window. KDE's clipboard history (Klipper) may hold on to the image, but I haven't checked that.

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
