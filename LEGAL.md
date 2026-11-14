# License, independence, trademarks and warranty

This file describes how Lumen relates to other software. It is a plain-language statement by the authors, not legal
advice. If you want to use Lumen commercially, or have a specific concern, ask the authors and have a lawyer review it.

## License

Lumen is licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE.md). You may use, change and share it for
any noncommercial purpose: personal use, research, teaching, and use by public research, educational, charitable and
government organisations. Commercial use needs a separate license from the authors.

## Relation to ImageJ and Fiji

* Lumen is a separate program, written in Python and TypeScript. It does not include ImageJ's or Fiji's Java code, jars
  or plugins.
* Many of its numeric routines were written to give the same results as **ImageJ 1.x**, which the NIH released into the
  public domain. These include the filters, background subtraction, binary operations, measurements, the
  auto-threshold methods and type conversions. Those Lumen modules are ports of ImageJ's published source code, and
  they name the ImageJ class each one follows. [docs/CROSSVALIDATION.md](docs/CROSSVALIDATION.md) records how closely
  they agree.
* The colour lookup tables reproduce the tables that ImageJ and Fiji ship, so that images look the same.
* Functions modelled on Fiji plugins (MorphoLibJ-style label analysis, TrackMate-style tracking, Weka-style pixel
  classification, bUnwarpJ-style registration, stitching and others) were written for Lumen from the plugins'
  publications, documentation and observable behaviour. Those plugins remain under their own licenses (mostly GPL or
  LGPL), and Lumen does not contain their code.
* The optional Java bridges run Fiji or ImageJ jars that **you** already have installed (Bio-Formats, the JVM script
  languages, plugins from Fiji update sites), in a separate Java process. Lumen does not ship those jars. Plugins you
  install from a Fiji update site keep their own licenses.

## Trademarks

ImageJ and Fiji are the names of their respective open-source projects. Other product names mentioned in this
repository (Bio-Formats, TrackMate, MorphoLibJ, Weka, Imaris, Zeiss, Nikon, Leica, Olympus, …) belong to their owners.
They are used only to describe compatibility. Lumen is **not affiliated with, endorsed by or sponsored by** the ImageJ or
Fiji projects, the NIH, or any maker of microscopes or imaging software. If a rights holder has a concern about a name or
description in this repository, please open an issue and it will be changed promptly.

## Warranty

Lumen is provided "as is", without warranty of any kind (see the license). It is research software, not a medical
device, and is not validated for diagnostic use. Whoever uses a result is responsible for checking it against their own
controls and data.

## AI assistants

The optional MCP server and the Agent Chat panel let AI assistants (Claude Code, Codex, Claude Desktop) work in a
running Lumen. They run on your computer and talk to Lumen only over `127.0.0.1`. What the assistant sends to its
provider is governed by that provider's terms, so do not let an assistant work on data you are not allowed to share
with it.
