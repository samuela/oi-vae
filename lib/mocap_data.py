import os

from lark import Lark, inline_args, Transformer


# This assumes that you're running experiments from the project root. Could be
# made more robust but a convenient assumption.
DATADIR = os.path.join(os.getcwd(), 'data')

amc_grammar = r"""
  start: preamble keyframe*

  preamble: ":FULLY-SPECIFIED" ":DEGREES"

  keyframe: INT (measurement)+
  measurement: WORD " " (SIGNED_NUMBER " ")* SIGNED_NUMBER

  COMMENT: "#" /[^\n]/* /\n/

  %import common.INT
  %import common.NEWLINE
  %import common.SIGNED_NUMBER
  %import common.WORD

  %ignore COMMENT
  %ignore NEWLINE
"""

class TreeToData(Transformer):
  def start(self, stuff):
    return stuff[1:]

  def keyframe(self, stuff):
    return int(stuff[0].value), dict(stuff[1:])

  def measurement(self, stuff):
    return stuff[0].value, [float(tok.value) for tok in stuff[1:]]

amc_parser = Lark(amc_grammar, parser='lalr', lexer='contextual')

def subject_trial_path(subject, trial, datadir=DATADIR):
  """Put together the path for the .amc file corresponding a particular trial by
  the given subject."""
  subject_str = '{:0>2d}'.format(subject)
  trial_str = '{:0>2d}'.format(trial)
  filename = '{}_{}.amc'.format(subject_str, trial_str)
  return os.path.join(
    datadir,
    'cmu_mocap',
    'subjects',
    subject_str,
    filename
  )

def parse_amc_data(raw_contents):
  """Parse the string contents of an .amc file into a list of data items. Each
  list element is a pair consisting of the key frame id and a dict mapping joint
  names to lists of channels (rotation, etc)."""
  results = amc_parser.parse(raw_contents)
  return TreeToData().transform(results)

def amc_to_array(parsed_amc, joint_order):
  """Take the parse .amc file contents from `parse_amc_data()` and put them into
  an "array" (technically just a list of lists; you can choose what to do with
  it)."""
  # Check that all time points are in sorted order and accounted for
  assert [i for i, _ in parsed_amc] == list(range(1, len(parsed_amc) + 1))

  # Check that the joint names are the same across all time points and have the
  # same dimensionality
  joint_dims = {joint: len(parsed_amc[0][1][joint]) for joint in joint_order}
  assert all([
    sorted(joint_order) == sorted(kvs.keys())
    for _, kvs in parsed_amc
  ])
  assert all([
    len(vs) == joint_dims[k]
    for _, kvs in parsed_amc
    for k, vs in kvs.items()
  ])

  arr = [
    [val for jn in joint_order for val in kvs[jn]]
    for _, kvs in parsed_amc
  ]

  return joint_dims, arr

def load_mocap_trial(subject, trial, joint_order=None, datadir=DATADIR):
  """Complete pipeline for loading subject/trial data into an array. The root
  joint is guaranteed to be the first in line for easier downstream processing.

  Returns
  =======
  joint_order : list of strings
  joint_dims : dict mapping joint names to its constituent number of channels
  arr : list of list of floats. The channel "array"
  """
  path = subject_trial_path(subject, trial, datadir=datadir)
  raw_contents = open(path, 'r').read()
  parsed = parse_amc_data(raw_contents)

  if joint_order == None:
    # Make sure to put root in front because we want to remove the first three
    # translation channels.
    joint_order = ['root'] + [k for k in parsed[0][1].keys() if k != 'root']

  joint_dims, arr = amc_to_array(parsed, joint_order)
  return joint_order, joint_dims, arr
