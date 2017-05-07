# tiled-dreamer

Deepdream for any size image.

I got annoyed with artists keeping their any-sized deep dream code closed source,
and making a big deal about it.

So made my own in a few hours - peow!

Make sure you have caffe installed and it's python modules in your `PYTHONPATH`

```
pip install -r requirements
python tiled_dreamer.py -i my_large_input.jpg -o my_large_output.jpg -t inception_4e/3x3
```

Also of interest might be the `--explore` option, which will iterate through the layers
and try dreaming on each (while filtering our the dummy "split" layers Caffe creates).

## How it works

To deepdream at any size:

- split into tiles, but overlap the tiles by N pixels(here N is hardcoded to 32), this ensures
  the edge pixels have the required context
- jitter each tile in the same direction. This means saving the jitter of the first tiled
  moving all tiles by the same amount.
- combine the tiles, by discarding the edges (half of the overlap amount), also deal with
  boundary conditions
- when you move up an octave, split into new tile sizes.

## License

MIT/BSD or just do whatever you like with it. No warranties, blah blah, etc.

Code adapted from the original deep dream ipython notebook.