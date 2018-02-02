from lark import Lark, inline_args, Transformer


raw_contents = open('/Users/samuelainsworth/Downloads/10_01.amc', 'r').read()

grammar = r"""
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

parser = Lark(grammar, parser='lalr', lexer='contextual')
results = parser.parse(raw_contents)
transformed = TreeToData().transform(results)

def amc_to_array(parsed_amc, joint_order):
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

joint_order = transformed[0][1].keys()
joint_dims, arr = amc_to_array(transformed, joint_order)

# def amc_to_array(parsed_amc):
#   # Check that all time points are in sorted order and accounted for
#   assert [i for i, _ in parsed_amc] == list(range(1, len(parsed_amc) + 1))

#   # Check that the joint names are the same across all time points and have the
#   # same dimensionality
#   joint_names = sorted(parsed_amc[0][1].keys())
#   joint_dims = {jn: len(parsed_amc[0][1][jn]) for jn in joint_names}
#   assert all([joint_names == sorted(kvs.keys()) for _, kvs in parsed_amc])
#   assert all([
#     len(vs) == joint_dims[k]
#     for _, kvs in parsed_amc
#     for k, vs in kvs.items()
#   ])

#   arr = [
#     [val for jn in joint_names for val in kvs[jn]]
#     for _, kvs in parsed_amc
#   ]

#   return joint_names, joint_dims, arr
