"""Fast-path synthetic data: beginner-level functions built from templates (families marked "near" only).

Every example is produced by this program (no language-model text), its expected outputs are computed by running
it inside the sandbox, and it is dropped if its name or its code fingerprint matches any benchmark task.
"""
import gzip
import json
import random
import sys
import textwrap

from lsllm.alpha import alpha_key
from lsllm.canon import canon
from lsllm.sandbox import run_many

P1 = [("x",), ("n",), ("num",), ("value",), ("number",)]
P2 = [("a", "b"), ("x", "y"), ("first", "second"), ("m", "n")]
P3 = [("a", "b", "c"), ("x", "y", "z"), ("first", "second", "third")]
L1 = [("nums",), ("numbers",), ("values",), ("items",), ("lst",), ("data",)]
S1 = [("s",), ("text",), ("word",), ("string",)]
W1 = [("words",), ("strings",), ("names",)]
IN_N = [("0",), ("1",), ("2",), ("5",), ("10",), ("-3",), ("7",), ("12",), ("-8",), ("20",)]
IN_NP = [("1",), ("2",), ("3",), ("5",), ("6",), ("10",), ("4",)]
IN_2 = [("1", "2"), ("5", "3"), ("0", "0"), ("10", "4"), ("-2", "7"), ("6", "6"), ("9", "2"), ("3", "8")]
IN_2NZ = [("1", "2"), ("6", "3"), ("10", "5"), ("7", "2"), ("-8", "4"), ("9", "3"), ("0", "5"), ("12", "4")]
IN_3 = [("1", "2", "3"), ("5", "5", "5"), ("0", "4", "2"), ("-1", "6", "3"), ("10", "2", "7"), ("4", "9", "4")]
IN_L = [("[1, 2, 3]",), ("[]",), ("[5]",), ("[4, -1, 0, 7]",), ("[2, 2, 6, 3]",), ("[-5, -2, 8, 10, 3]",), ("[0, 0, 1]",), ("[9, 4, 6, 1, 12]",)]
IN_LNE = [g for g in IN_L if g != ("[]",)]
IN_S = [("'hello'",), ("''",), ("'a'",), ("'Hello World'",), ("'abc123'",), ("'Python'",), ("'UPPER case'",), ("'42'",)]
IN_SNE = [g for g in IN_S if g != ("''",)]
IN_W = [("['a', 'bb', 'ccc']",), ("[]",), ("['hello']",), ("['Hi', 'there', 'you']",), ("['x', '', 'yz']",)]
IN_B2 = [("True", "True"), ("True", "False"), ("False", "True"), ("False", "False")]
IN_B3 = [("True", "True", "True"), ("True", "False", "True"), ("False", "False", "False"), ("False", "True", "False")]
WORD = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}

E = []


def fam(family, names, args, docs, bodies, inputs):
    E.append((family, names, args, docs, bodies, inputs))


def loop(init, step, result="total"):
    return "%s\nfor item in {0}:\n    %s\nreturn %s" % (init, step, result)


# ---- one number
for k in [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 25, 50, 100]:
    fam("arith_one", ["add_%d" % k, "plus_%d" % k, "increase_by_%d" % k], P1, ["Return {0} plus %d." % k, "Return the number {0} increased by %d." % k], ["return {0} + %d" % k], IN_N)
    fam("arith_one", ["subtract_%d" % k, "minus_%d" % k, "decrease_by_%d" % k], P1, ["Return {0} minus %d." % k, "Return the number {0} decreased by %d." % k], ["return {0} - %d" % k], IN_N)
    fam("arith_one", ["times_%d" % k, "multiply_by_%d" % k], P1, ["Return {0} times %d." % k, "Return the number {0} multiplied by %d." % k], ["return {0} * %d" % k], IN_N)
    fam("arith_one", ["remainder_by_%d" % k, "mod_%d" % k], P1, ["Return the remainder of {0} divided by %d." % k], ["return {0} %% %d" % k], IN_N)
    fam("arith_one", ["whole_divide_by_%d" % k, "floor_divide_by_%d" % k], P1, ["Return {0} divided by %d, rounded down to a whole number." % k], ["return {0} // %d" % k], IN_N)
    fam("parity_sign", ["is_divisible_by_%d" % k, "is_multiple_of_%d" % k, "divides_by_%d" % k], P1, ["Return True if {0} is divisible by %d, otherwise False." % k, "Return True if {0} is a multiple of %d." % k], ["return {0} %% %d == 0" % k], IN_N + [(str(k),), (str(3 * k),)])
    fam("compare", ["is_greater_than_%d" % k, "is_above_%d" % k, "more_than_%d" % k], P1, ["Return True if {0} is greater than %d, otherwise False." % k], ["return {0} > %d" % k], IN_N + [(str(k),), (str(k + 1),)])
    fam("compare", ["is_less_than_%d" % k, "is_below_%d" % k], P1, ["Return True if {0} is less than %d, otherwise False." % k], ["return {0} < %d" % k], IN_N + [(str(k),), (str(k - 1),)])
    fam("compare", ["is_at_least_%d" % k], P1, ["Return True if {0} is %d or more, otherwise False." % k], ["return {0} >= %d" % k], IN_N + [(str(k),)])
    fam("compare", ["is_exactly_%d" % k, "equals_%d" % k], P1, ["Return True if {0} is equal to %d, otherwise False." % k], ["return {0} == %d" % k], IN_N + [(str(k),)])
    fam("conditional_value", ["at_most_%d" % k, "cap_at_%d" % k, "limit_to_%d" % k], P1, ["Return {0}, but never more than %d." % k], ["if {0} > %d:\n    return %d\nreturn {0}" % (k, k), "return min({0}, %d)" % k], IN_N + [(str(k + 5),)])
    fam("conditional_value", ["at_least_%d" % k, "raise_to_%d" % k], P1, ["Return {0}, but never less than %d." % k], ["if {0} < %d:\n    return %d\nreturn {0}" % (k, k), "return max({0}, %d)" % k], IN_N + [(str(k + 5),)])
for a in range(2, 6):
    for b in range(1, 6):
        fam("arith_one", ["times_%d_plus_%d" % (a, b)], P1, ["Return {0} times %d plus %d." % (a, b)], ["return {0} * %d + %d" % (a, b)], IN_N)
        fam("arith_one", ["times_%d_minus_%d" % (a, b)], P1, ["Return {0} times %d minus %d." % (a, b)], ["return {0} * %d - %d" % (a, b)], IN_N)
        fam("arith_one", ["plus_%d_times_%d" % (b, a)], P1, ["Return the sum of {0} and %d, multiplied by %d." % (b, a)], ["return ({0} + %d) * %d" % (b, a)], IN_N)
for k in [2, 3, 4, 5]:
    fam("arith_one", ["power_%d" % k, "to_the_power_%d" % k, "raise_to_%d" % k], P1, ["Return {0} raised to the power %d." % k], ["return {0} ** %d" % k], IN_N)
fam("arith_one", ["double", "twice", "times_two", "double_value"], P1, ["Return {0} multiplied by 2.", "Return twice the number {0}."], ["return {0} * 2", "return 2 * {0}", "return {0} + {0}"], IN_N)
fam("arith_one", ["triple", "times_three", "triple_value"], P1, ["Return {0} multiplied by 3.", "Return three times the number {0}."], ["return {0} * 3", "return 3 * {0}"], IN_N)
fam("arith_one", ["square", "squared", "square_of"], P1, ["Return the square of {0}.", "Return {0} multiplied by itself."], ["return {0} * {0}", "return {0} ** 2"], IN_N)
fam("arith_one", ["cube", "cubed", "cube_of"], P1, ["Return the cube of {0}.", "Return {0} raised to the power 3."], ["return {0} ** 3", "return {0} * {0} * {0}"], IN_N)
fam("arith_one", ["half", "halve", "half_of"], P1, ["Return half of {0}.", "Return {0} divided by 2."], ["return {0} / 2"], IN_N)
fam("arith_one", ["negate", "opposite", "change_sign", "minus"], P1, ["Return {0} with the opposite sign.", "Return the negative of {0}."], ["return -{0}"], IN_N)
fam("arith_one", ["increment", "add_one", "plus_one", "next_number", "successor"], P1, ["Return {0} plus 1.", "Return the number that follows {0}."], ["return {0} + 1"], IN_N)
fam("arith_one", ["decrement", "subtract_one", "minus_one", "previous_number"], P1, ["Return {0} minus 1.", "Return the number just before {0}."], ["return {0} - 1"], IN_N)
fam("arith_one", ["identity", "same_value", "keep"], P1, ["Return {0} unchanged."], ["return {0}"], IN_N)
fam("parity_sign", ["is_odd_number", "odd", "is_not_even"], P1, ["Return True if {0} is odd, otherwise False."], ["return {0} % 2 == 1", "return {0} % 2 != 0"], IN_N)
fam("parity_sign", ["is_even_number", "even", "is_not_odd"], P1, ["Return True if {0} is even, otherwise False."], ["return {0} % 2 == 0"], IN_N)
fam("parity_sign", ["is_positive", "positive", "above_zero", "is_greater_than_zero"], P1, ["Return True if {0} is greater than zero, otherwise False."], ["return {0} > 0"], IN_N)
fam("parity_sign", ["is_negative", "negative", "below_zero", "is_less_than_zero"], P1, ["Return True if {0} is less than zero, otherwise False."], ["return {0} < 0"], IN_N)
fam("parity_sign", ["is_zero", "equals_zero", "is_nothing"], P1, ["Return True if {0} is zero, otherwise False."], ["return {0} == 0"], IN_N)
fam("parity_sign", ["is_not_zero", "is_nonzero", "non_zero"], P1, ["Return True if {0} is not zero, otherwise False."], ["return {0} != 0"], IN_N)
fam("parity_sign", ["is_not_negative", "zero_or_more", "is_non_negative"], P1, ["Return True if {0} is zero or greater, otherwise False."], ["return {0} >= 0"], IN_N)
fam("parity_sign", ["is_not_positive", "zero_or_less"], P1, ["Return True if {0} is zero or less, otherwise False."], ["return {0} <= 0"], IN_N)
fam("conditional_value", ["absolute", "absolute_value", "magnitude", "size_of"], P1, ["Return the absolute value of {0}.", "Return {0} without its sign."], ["if {0} < 0:\n    return -{0}\nreturn {0}", "return abs({0})", "if {0} >= 0:\n    return {0}\nelse:\n    return -{0}"], IN_N)
fam("conditional_value", ["never_negative", "zero_if_negative", "floor_at_zero", "clip_at_zero"], P1, ["Return {0}, or 0 if {0} is negative."], ["if {0} < 0:\n    return 0\nreturn {0}", "return max({0}, 0)"], IN_N)
fam("conditional_value", ["sign", "sign_of", "signum"], P1, ["Return 1 if {0} is positive, -1 if it is negative and 0 if it is zero."], ["if {0} > 0:\n    return 1\nif {0} < 0:\n    return -1\nreturn 0"], IN_N)
fam("conditional_value", ["zero_if_odd", "keep_if_even"], P1, ["Return {0} if it is even, otherwise 0."], ["if {0} % 2 == 0:\n    return {0}\nreturn 0"], IN_N)
fam("conditional_value", ["double_if_positive"], P1, ["Return {0} times 2 if {0} is positive, otherwise {0}."], ["if {0} > 0:\n    return {0} * 2\nreturn {0}"], IN_N)
for age, low, high, what in [(12, 5, 10, "ticket_price"), (18, 3, 8, "entry_fee"), (65, 20, 12, "bus_fare"), (10, 2, 6, "museum_price"), (16, 7, 15, "pool_price")]:
    fam("conditional_value", [what, what + "_by_age"], [("age",), ("years",)], ["Return %d if {0} is less than %d, otherwise %d." % (low, age, high)], ["if {0} < %d:\n    return %d\nreturn %d" % (age, low, high), "if {0} < %d:\n    return %d\nelse:\n    return %d" % (age, low, high)], [("5",), ("30",), (str(age),), (str(age - 1),), ("70",)])

# ---- two and three numbers
BIN = [(["add", "sum_two", "plus", "add_numbers", "total_of_two"], "the sum of {0} and {1}", "{0} + {1}", IN_2),
       (["subtract", "difference", "minus_two", "take_away"], "{0} minus {1}", "{0} - {1}", IN_2),
       (["multiply", "product", "times", "multiply_numbers"], "the product of {0} and {1}", "{0} * {1}", IN_2),
       (["divide", "quotient", "divide_numbers"], "{0} divided by {1}", "{0} / {1}", [("6", "3"), ("10", "5"), ("8", "2"), ("9", "3"), ("0", "4")]),
       (["whole_quotient", "floor_divide", "integer_divide"], "{0} divided by {1}, rounded down to a whole number", "{0} // {1}", IN_2NZ),
       (["remainder", "modulo", "remainder_of"], "the remainder of {0} divided by {1}", "{0} % {1}", IN_2NZ),
       (["power", "raise_to_power", "to_the_power"], "{0} raised to the power {1}", "{0} ** {1}", [("2", "3"), ("5", "2"), ("3", "0"), ("10", "1"), ("4", "2")]),
       (["average_of_two", "mean_of_two", "midpoint"], "the average of {0} and {1}", "({0} + {1}) / 2", [("2", "4"), ("0", "0"), ("10", "20"), ("-2", "2"), ("7", "9")]),
       (["absolute_difference", "distance_between", "gap"], "the absolute difference between {0} and {1}", "abs({0} - {1})", IN_2),
       (["sum_of_squares_two", "squares_sum"], "the sum of the squares of {0} and {1}", "{0} * {0} + {1} * {1}", IN_2),
       (["double_sum", "twice_the_sum"], "two times the sum of {0} and {1}", "({0} + {1}) * 2", IN_2),
       (["square_of_sum"], "the square of the sum of {0} and {1}", "({0} + {1}) ** 2", IN_2),
       (["first_plus_double_second"], "{0} plus two times {1}", "{0} + 2 * {1}", IN_2),
       (["difference_of_squares"], "the square of {0} minus the square of {1}", "{0} * {0} - {1} * {1}", IN_2),
       (["sum_plus_one"], "the sum of {0} and {1}, plus 1", "{0} + {1} + 1", IN_2),
       (["product_minus_sum"], "the product of {0} and {1} minus their sum", "{0} * {1} - ({0} + {1})", IN_2)]
for names, doc, expr, inputs in BIN:
    fam("arith_two", names, P2, ["Return %s." % doc], ["return " + expr, "result = %s\nreturn result" % expr], inputs)
TRI = [(["sum_of_three", "add_three", "total_of_three"], "the sum of {0}, {1} and {2}", "{0} + {1} + {2}"),
       (["product_of_three", "multiply_three"], "the product of {0}, {1} and {2}", "{0} * {1} * {2}"),
       (["average_of_three", "mean_of_three"], "the average of {0}, {1} and {2}", "({0} + {1} + {2}) / 3"),
       (["sum_times_third"], "the sum of {0} and {1}, multiplied by {2}", "({0} + {1}) * {2}"),
       (["product_plus_third"], "the product of {0} and {1}, plus {2}", "{0} * {1} + {2}"),
       (["first_minus_others"], "{0} minus {1} minus {2}", "{0} - {1} - {2}"),
       (["first_plus_product"], "{0} plus the product of {1} and {2}", "{0} + {1} * {2}"),
       (["sum_of_three_squares"], "the sum of the squares of {0}, {1} and {2}", "{0} ** 2 + {1} ** 2 + {2} ** 2"),
       (["double_sum_of_three"], "two times the sum of {0}, {1} and {2}", "2 * ({0} + {1} + {2})")]
for names, doc, expr in TRI:
    fam("arith_three", names, P3, ["Return %s." % doc], ["return " + expr, "total = %s\nreturn total" % expr], [("3", "6", "9"), ("0", "0", "0"), ("1", "2", "3"), ("-3", "6", "3"), ("12", "3", "6")])
FORMULA = [("rectangle_area", ("width", "height"), "the area of a rectangle with the given width and height", "width * height"),
           ("rectangle_perimeter", ("width", "height"), "the perimeter of a rectangle with the given width and height", "2 * (width + height)"),
           ("square_area", ("side",), "the area of a square with the given side", "side * side"),
           ("square_perimeter", ("side",), "the perimeter of a square with the given side", "4 * side"),
           ("box_volume", ("length", "width", "height"), "the volume of a box with the given length, width and height", "length * width * height"),
           ("triangle_area", ("base", "height"), "the area of a triangle, which is base times height divided by 2", "base * height / 2"),
           ("total_price", ("price", "quantity"), "the total price of quantity items that cost price each", "price * quantity"),
           ("total_cost", ("price", "count", "shipping"), "price times count plus shipping", "price * count + shipping"),
           ("average_speed", ("distance", "time"), "the distance divided by the time", "distance / time"),
           ("distance_travelled", ("speed", "time"), "the speed multiplied by the time", "speed * time"),
           ("change_due", ("paid", "price"), "the amount paid minus the price", "paid - price"),
           ("points_total", ("wins", "draws"), "3 points for every win plus 1 point for every draw", "wins * 3 + draws"),
           ("legs_count", ("chickens", "cows"), "the number of legs of the chickens (2 each) and the cows (4 each)", "chickens * 2 + cows * 4"),
           ("pages_left", ("total", "read"), "the number of pages left when read pages of total are done", "total - read"),
           ("cube_volume", ("side",), "the volume of a cube with the given side", "side ** 3"),
           ("simple_interest", ("amount", "rate", "years"), "amount times rate times years divided by 100", "amount * rate * years / 100"),
           ("weekly_pay", ("hours", "rate"), "the pay for the given hours at the given rate per hour", "hours * rate"),
           ("items_per_box", ("items", "boxes"), "how many whole items fit in each box when items are shared between boxes", "items // boxes"),
           ("items_left_over", ("items", "boxes"), "how many items are left over when items are shared equally between boxes", "items % boxes")]
for name, args, doc, expr in FORMULA:
    ins = {1: [("3",), ("10",), ("1",), ("6",)], 2: [("4", "2"), ("10", "5"), ("6", "3"), ("8", "4")], 3: [("2", "3", "4"), ("10", "5", "2"), ("100", "5", "2"), ("6", "1", "3")]}[len(args)]
    fam("arith_three", [name, "get_" + name, "compute_" + name], [args], ["Return %s." % doc], ["return " + expr], ins)
CMP = [(["is_greater", "is_larger", "is_bigger", "greater_than"], "{0} is greater than {1}", "{0} > {1}"),
       (["is_less", "is_smaller", "less_than"], "{0} is less than {1}", "{0} < {1}"),
       (["is_at_least", "not_less", "greater_or_equal"], "{0} is greater than or equal to {1}", "{0} >= {1}"),
       (["is_at_most", "not_greater", "less_or_equal"], "{0} is less than or equal to {1}", "{0} <= {1}"),
       (["are_equal", "same_value_pair", "equal"], "{0} and {1} are equal", "{0} == {1}"),
       (["are_different", "not_equal", "differ"], "{0} and {1} are not equal", "{0} != {1}"),
       (["sum_is_positive"], "the sum of {0} and {1} is greater than zero", "{0} + {1} > 0"),
       (["is_double_of"], "{0} is two times {1}", "{0} == 2 * {1}"),
       (["is_divisible"], "{0} is divisible by {1}", "{0} % {1} == 0"),
       (["same_sign"], "{0} and {1} are both positive or both negative", "{0} * {1} > 0"),
       (["sum_is_even"], "the sum of {0} and {1} is even", "({0} + {1}) % 2 == 0")]
for names, doc, expr in CMP:
    fam("compare", names, P2, ["Return True if %s, otherwise False." % doc], ["return " + expr], IN_2NZ)
fam("compare", ["is_between", "in_range", "is_within"], [("x", "low", "high"), ("value", "lowest", "highest"), ("n", "start", "end")], ["Return True if {0} is at least {1} and at most {2}, otherwise False."], ["return {1} <= {0} <= {2}", "return {0} >= {1} and {0} <= {2}"], [("5", "1", "10"), ("0", "1", "10"), ("10", "1", "10"), ("11", "1", "10"), ("3", "3", "3")])
fam("compare", ["all_equal", "all_same", "three_equal"], P3, ["Return True if {0}, {1} and {2} are all equal, otherwise False."], ["return {0} == {1} == {2}", "return {0} == {1} and {1} == {2}"], IN_3)
fam("compare", ["is_increasing", "in_order", "ascending"], P3, ["Return True if {0} is less than {1} and {1} is less than {2}, otherwise False."], ["return {0} < {1} < {2}", "return {0} < {1} and {1} < {2}"], IN_3)
fam("min_max_few", ["larger", "bigger", "max_of_two", "larger_of_two", "maximum_of_two"], P2, ["Return the larger of {0} and {1}."], ["if {0} > {1}:\n    return {0}\nreturn {1}", "return max({0}, {1})", "if {0} >= {1}:\n    return {0}\nelse:\n    return {1}"], IN_2)
fam("min_max_few", ["smaller", "lesser", "min_of_two", "smaller_of_two", "minimum_of_two"], P2, ["Return the smaller of {0} and {1}."], ["if {0} < {1}:\n    return {0}\nreturn {1}", "return min({0}, {1})"], IN_2)
fam("min_max_few", ["largest_of_three", "max_of_three", "biggest_of_three"], P3, ["Return the largest of {0}, {1} and {2}."], ["return max({0}, {1}, {2})", "largest = {0}\nif {1} > largest:\n    largest = {1}\nif {2} > largest:\n    largest = {2}\nreturn largest"], IN_3)
fam("min_max_few", ["smallest_of_three", "min_of_three", "least_of_three"], P3, ["Return the smallest of {0}, {1} and {2}."], ["return min({0}, {1}, {2})", "smallest = {0}\nif {1} < smallest:\n    smallest = {1}\nif {2} < smallest:\n    smallest = {2}\nreturn smallest"], IN_3)
fam("min_max_few", ["spread_of_three", "range_of_three"], P3, ["Return the largest of {0}, {1} and {2} minus the smallest of them."], ["return max({0}, {1}, {2}) - min({0}, {1}, {2})"], IN_3)
fam("min_max_few", ["larger_minus_smaller"], P2, ["Return the larger of {0} and {1} minus the smaller one."], ["return max({0}, {1}) - min({0}, {1})"], IN_2)
fam("min_max_few", ["double_the_larger"], P2, ["Return two times the larger of {0} and {1}."], ["return max({0}, {1}) * 2"], IN_2)
fam("min_max_few", ["sum_of_extremes"], P3, ["Return the sum of the largest and the smallest of {0}, {1} and {2}."], ["return max({0}, {1}, {2}) + min({0}, {1}, {2})"], IN_3)
B2 = [("a", "b"), ("p", "q"), ("first", "second"), ("x", "y")]
BOOL = [(["both", "both_true", "logical_and"], "both {0} and {1} are True", "{0} and {1}"),
        (["either", "any_true", "logical_or"], "{0} or {1} is True", "{0} or {1}"),
        (["neither", "none_true", "both_false"], "neither {0} nor {1} is True", "not {0} and not {1}"),
        (["exactly_one", "only_one", "one_but_not_both"], "exactly one of {0} and {1} is True", "{0} != {1}"),
        (["not_both", "at_most_one"], "{0} and {1} are not both True", "not ({0} and {1})"),
        (["same_truth", "both_or_neither"], "{0} and {1} have the same truth value", "{0} == {1}"),
        (["first_only", "first_not_second"], "{0} is True and {1} is False", "{0} and not {1}"),
        (["second_only"], "{1} is True and {0} is False", "{1} and not {0}")]
for names, doc, expr in BOOL:
    fam("boolean_logic", names, B2, ["Return True if %s, otherwise False." % doc], ["return " + expr], IN_B2)
fam("boolean_logic", ["invert", "logical_not", "flip"], [("flag",), ("value",), ("a",)], ["Return True if {0} is False, and False if {0} is True."], ["return not {0}"], [("True",), ("False",)])
fam("boolean_logic", ["all_three_true", "all_of_three"], P3, ["Return True if {0}, {1} and {2} are all True, otherwise False."], ["return {0} and {1} and {2}"], IN_B3)
fam("boolean_logic", ["any_of_three", "at_least_one_true"], P3, ["Return True if at least one of {0}, {1} and {2} is True, otherwise False."], ["return {0} or {1} or {2}"], IN_B3)
fam("boolean_logic", ["none_of_three"], P3, ["Return True if {0}, {1} and {2} are all False, otherwise False."], ["return not ({0} or {1} or {2})"], IN_B3)

# ---- unit conversions
UNITS = [("hours", "minutes", "* 60"), ("days", "hours", "* 24"), ("weeks", "days", "* 7"), ("years", "months", "* 12"), ("hours", "seconds", "* 3600"),
         ("kilometers", "meters", "* 1000"), ("meters", "centimeters", "* 100"), ("centimeters", "millimeters", "* 10"), ("kilograms", "grams", "* 1000"),
         ("liters", "milliliters", "* 1000"), ("dollars", "cents", "* 100"), ("feet", "inches", "* 12"), ("yards", "feet", "* 3"), ("bytes", "bits", "* 8"),
         ("dozens", "items", "* 12"), ("pounds", "ounces", "* 16"), ("minutes", "hours", "/ 60"), ("grams", "kilograms", "/ 1000"), ("meters", "kilometers", "/ 1000"),
         ("cents", "dollars", "/ 100"), ("months", "years", "// 12"), ("days", "weeks", "// 7"), ("seconds", "minutes", "// 60"), ("inches", "feet", "// 12"),
         ("decades", "years", "* 10"), ("centuries", "years", "* 100"), ("tons", "kilograms", "* 1000"), ("euros", "cents", "* 100"), ("kilobytes", "bytes", "* 1024")]
for src, dst, op in UNITS:
    whole = "whole " if op.startswith("//") else ""
    fam("unit_conversion", ["%s_to_%s" % (src, dst), "convert_%s_to_%s" % (src, dst), "%s_in_%s" % (dst, src)], [(src,), ("value",), ("amount",)],
        ["Return the number of %s%s in the given number of %s." % (whole, dst, src), "Convert {0} from %s to %s%s." % (src, whole, dst)], ["return {0} %s" % op], [("1",), ("0",), ("120",), ("600",), ("2400",), ("36",)])

# ---- loops over a range
RL = [(["sum_to", "sum_up_to", "sum_first_numbers", "triangle_number"], "the sum of the whole numbers from 1 to {0}", "total = 0\nfor k in range(1, {0} + 1):\n    total += k\nreturn total", "return sum(range(1, {0} + 1))"),
      (["sum_of_squares_to", "sum_squares_up_to"], "the sum of the squares of the whole numbers from 1 to {0}", "total = 0\nfor k in range(1, {0} + 1):\n    total += k * k\nreturn total", "return sum((k * k for k in range(1, {0} + 1)))"),
      (["product_to", "factorial", "product_up_to"], "the product of the whole numbers from 1 to {0}", "result = 1\nfor k in range(1, {0} + 1):\n    result *= k\nreturn result", None),
      (["sum_even_to", "sum_of_evens_up_to"], "the sum of the even numbers from 1 to {0}", "total = 0\nfor k in range(1, {0} + 1):\n    if k % 2 == 0:\n        total += k\nreturn total", "return sum(range(2, {0} + 1, 2))"),
      (["sum_odd_to", "sum_of_odds_up_to"], "the sum of the odd numbers from 1 to {0}", "total = 0\nfor k in range(1, {0} + 1):\n    if k % 2 == 1:\n        total += k\nreturn total", "return sum(range(1, {0} + 1, 2))"),
      (["count_evens_to", "evens_up_to_count"], "how many even numbers there are from 1 to {0}", "count = 0\nfor k in range(1, {0} + 1):\n    if k % 2 == 0:\n        count += 1\nreturn count", None),
      (["sum_of_cubes_to"], "the sum of the cubes of the whole numbers from 1 to {0}", "total = 0\nfor k in range(1, {0} + 1):\n    total += k ** 3\nreturn total", None),
      (["sum_below"], "the sum of the whole numbers from 0 to {0} - 1", "total = 0\nfor k in range({0}):\n    total += k\nreturn total", "return sum(range({0}))"),
      (["sum_of_doubles_to"], "the sum of two times every whole number from 1 to {0}", "total = 0\nfor k in range(1, {0} + 1):\n    total += 2 * k\nreturn total", None)]
for names, doc, body, alt in RL:
    fam("range_loops", names, [("n",), ("limit",), ("number",), ("last",)], ["Return %s." % doc], [body] + ([alt] if alt else []), IN_NP + [("0",)])
for k in [3, 4, 5, 7, 10]:
    fam("range_loops", ["count_multiples_of_%d" % k, "multiples_of_%d_up_to" % k], [("n",), ("limit",)], ["Return how many numbers from 1 to {0} are divisible by %d." % k],
        ["count = 0\nfor g in range(1, {0} + 1):\n    if g %% %d == 0:\n        count += 1\nreturn count" % k, "return {0} // %d" % k], [("10",), ("0",), ("21",), ("35",), ("4",)])
    fam("range_loops", ["sum_multiples_of_%d" % k], [("n",), ("limit",)], ["Return the sum of the numbers from 1 to {0} that are divisible by %d." % k],
        ["total = 0\nfor g in range(1, {0} + 1):\n    if g %% %d == 0:\n        total += g\nreturn total" % k], [("10",), ("0",), ("21",), ("35",), ("4",)])
fam("range_loops", ["sum_between", "sum_range", "sum_from_to"], [("start", "end"), ("low", "high"), ("a", "b")], ["Return the sum of the whole numbers from {0} to {1}, both included."],
    ["total = 0\nfor k in range({0}, {1} + 1):\n    total += k\nreturn total", "return sum(range({0}, {1} + 1))"], [("1", "5"), ("3", "3"), ("0", "10"), ("-2", "2"), ("4", "7")])
fam("range_loops", ["product_between", "product_range"], [("start", "end"), ("low", "high")], ["Return the product of the whole numbers from {0} to {1}, both included."],
    ["result = 1\nfor k in range({0}, {1} + 1):\n    result *= k\nreturn result"], [("1", "5"), ("3", "3"), ("2", "4"), ("5", "6")])
fam("range_loops", ["count_between"], [("start", "end"), ("low", "high")], ["Return how many whole numbers there are from {0} to {1}, both included."], ["return {1} - {0} + 1"], [("1", "5"), ("3", "3"), ("0", "10"), ("-2", "2")])

# ---- build lists
LB = [(["numbers_up_to", "count_up", "one_to_n", "first_numbers"], "the list of whole numbers from 1 to {0}", ["return list(range(1, {0} + 1))", "result = []\nfor k in range(1, {0} + 1):\n    result.append(k)\nreturn result"]),
      (["numbers_below", "zero_to_n", "range_list"], "the list of whole numbers from 0 to {0} - 1", ["return list(range({0}))"]),
      (["countdown", "count_down", "n_to_one"], "the list of whole numbers from {0} down to 1", ["return list(range({0}, 0, -1))"]),
      (["even_numbers_up_to", "evens_to"], "the list of even numbers from 2 to {0}", ["return list(range(2, {0} + 1, 2))", "return [k for k in range(1, {0} + 1) if k % 2 == 0]"]),
      (["odd_numbers_up_to", "odds_to"], "the list of odd numbers from 1 to {0}", ["return list(range(1, {0} + 1, 2))", "return [k for k in range(1, {0} + 1) if k % 2 == 1]"]),
      (["squares_up_to", "first_squares"], "the list of the squares of the whole numbers from 1 to {0}", ["return [k * k for k in range(1, {0} + 1)]"]),
      (["cubes_up_to"], "the list of the cubes of the whole numbers from 1 to {0}", ["return [k ** 3 for k in range(1, {0} + 1)]"]),
      (["doubles_up_to"], "the list of two times every whole number from 1 to {0}", ["return [2 * k for k in range(1, {0} + 1)]"]),
      (["zeros", "list_of_zeros", "make_zeros"], "a list of {0} zeros", ["return [0] * {0}"]),
      (["ones", "list_of_ones"], "a list of {0} ones", ["return [1] * {0}"])]
for names, doc, bodies in LB:
    fam("list_build", names, [("n",), ("count",), ("size",), ("limit",)], ["Return %s." % doc], bodies, [("3",), ("0",), ("1",), ("5",), ("6",)])
for k in [2, 3, 4, 5, 10]:
    fam("list_build", ["multiples_of_%d" % k, "first_multiples_of_%d" % k], [("n",), ("count",)], ["Return the list of the first {0} multiples of %d, starting with %d." % (k, k)], ["return [%d * g for g in range(1, {0} + 1)]" % k], [("3",), ("0",), ("1",), ("5",)])
fam("list_build", ["repeat_value", "copies", "repeat_item", "fill_list"], [("value", "n"), ("item", "count"), ("x", "times")], ["Return a list with {1} copies of {0}."], ["return [{0}] * {1}"], [("7", "3"), ("0", "0"), ("'a'", "2"), ("5", "1")])
fam("list_build", ["range_between", "numbers_from_to", "list_from_to"], [("start", "end"), ("low", "high"), ("a", "b")], ["Return the list of whole numbers from {0} to {1}, both included."], ["return list(range({0}, {1} + 1))"], [("1", "4"), ("3", "3"), ("-1", "2"), ("5", "8")])
fam("list_build", ["pair", "make_pair", "two_item_list"], P2, ["Return a list with the two items {0} and {1}."], ["return [{0}, {1}]"], IN_2)
fam("list_build", ["triple_list", "three_item_list"], P3, ["Return a list with the three items {0}, {1} and {2}."], ["return [{0}, {1}, {2}]"], IN_3)

# ---- lists: aggregate, filter, map, predicates
PRED = [("even", "{x} % 2 == 0", "even"), ("odd", "{x} % 2 == 1", "odd"), ("positive", "{x} > 0", "greater than zero"), ("negative", "{x} < 0", "less than zero"),
        ("nonzero", "{x} != 0", "not zero"), ("zero", "{x} == 0", "equal to zero")]
for k in [1, 2, 3, 5, 10]:
    PRED += [("above_%d" % k, "{x} > %d" % k, "greater than %d" % k), ("below_%d" % k, "{x} < %d" % k, "less than %d" % k)]
for k in [3, 5]:
    PRED += [("divisible_by_%d" % k, "{x} %% %d == 0" % k, "divisible by %d" % k)]
MAPS = [("double", "{x} * 2", "multiplied by 2"), ("triple", "{x} * 3", "multiplied by 3"), ("square", "{x} * {x}", "squared"), ("negate", "-{x}", "with the opposite sign"),
        ("absolute", "abs({x})", "replaced by its absolute value"), ("increment", "{x} + 1", "increased by 1"), ("decrement", "{x} - 1", "decreased by 1"), ("cube", "{x} ** 3", "raised to the power 3")]
for k in [2, 5, 10]:
    MAPS += [("add_%d_to" % k, "{x} + %d" % k, "increased by %d" % k), ("times_%d" % k if k != 2 else "times_two", "{x} * %d" % k, "multiplied by %d" % k)]
VAR = ["x", "item", "n", "value"]
for pn, pe, pd in PRED:
    for v in VAR[:2]:
        test = pe.format(x=v)
        fam("list_filter", ["keep_%s" % pn, "%s_only" % pn, "filter_%s" % pn, "select_%s" % pn, "%s_items" % pn], L1, ["Return a list with the items of {0} that are %s." % pd, "Return the numbers of the list {0} that are %s, in the same order." % pd],
            ["return [%s for %s in {0} if %s]" % (v, v, test), "result = []\nfor %s in {0}:\n    if %s:\n        result.append(%s)\nreturn result" % (v, test, v)], IN_L)
        fam("list_aggregate", ["count_%s" % pn, "how_many_%s" % pn, "number_of_%s" % pn], L1, ["Return how many items of the list {0} are %s." % pd],
            ["return sum((1 for %s in {0} if %s))" % (v, test), "count = 0\nfor %s in {0}:\n    if %s:\n        count += 1\nreturn count" % (v, test), "return len([%s for %s in {0} if %s])" % (v, v, test)], IN_L)
        fam("list_aggregate", ["sum_%s" % pn, "sum_of_%s" % pn, "total_of_%s" % pn], L1, ["Return the sum of the items of the list {0} that are %s." % pd],
            ["return sum((%s for %s in {0} if %s))" % (v, v, test), "total = 0\nfor %s in {0}:\n    if %s:\n        total += %s\nreturn total" % (v, test, v)], IN_L)
        fam("list_predicates", ["all_%s" % pn, "every_item_%s" % pn, "are_all_%s" % pn], L1, ["Return True if every item of the list {0} is %s, otherwise False." % pd],
            ["return all((%s for %s in {0}))" % (test, v), "for %s in {0}:\n    if not %s:\n        return False\nreturn True" % (v, test)], IN_L)
        fam("list_predicates", ["any_%s" % pn, "has_%s" % pn, "contains_%s" % pn], L1, ["Return True if at least one item of the list {0} is %s, otherwise False." % pd],
            ["return any((%s for %s in {0}))" % (test, v), "for %s in {0}:\n    if %s:\n        return True\nreturn False" % (v, test)], IN_L)
for mn, me, md in MAPS:
    for v in VAR[:2]:
        expr = me.format(x=v)
        fam("list_map", ["%s_all" % mn, "%s_each" % mn, "%s_items" % mn, "%s_list" % mn], L1, ["Return a list with every item of {0} %s." % md, "Return a list with every number of the list {0} %s." % md],
            ["return [%s for %s in {0}]" % (expr, v), "result = []\nfor %s in {0}:\n    result.append(%s)\nreturn result" % (v, expr)], IN_L)
        fam("list_aggregate", ["sum_of_%s" % mn, "total_%s" % mn], L1, ["Return the sum of the items of the list {0}, each %s." % md],
            ["return sum((%s for %s in {0}))" % (expr, v), "total = 0\nfor %s in {0}:\n    total += %s\nreturn total" % (v, expr)], IN_L)
    for pn, pe, pd in PRED[:6]:
        fam("list_map", ["%s_the_%s" % (mn, pn)], L1, ["Return a list with the items of {0} that are %s, each %s." % (pd, md)], ["return [%s for x in {0} if %s]" % (me.format(x="x"), pe.format(x="x"))], IN_L)
AGG = [(["total", "sum_list", "sum_all", "list_sum", "add_all"], "the sum of all items of the list {0}", ["return sum({0})", loop("total = 0", "total += item")], IN_L),
       (["length", "count_items", "list_length", "size", "number_of_items"], "the number of items in the list {0}", ["return len({0})", loop("count = 0", "count += 1", "count")], IN_L),
       (["largest", "maximum", "biggest", "list_max", "highest"], "the largest item of the list {0}, which is not empty", ["return max({0})", "best = {0}[0]\nfor item in {0}:\n    if item > best:\n        best = item\nreturn best"], IN_LNE),
       (["smallest", "minimum", "list_min", "lowest", "least"], "the smallest item of the list {0}, which is not empty", ["return min({0})", "best = {0}[0]\nfor item in {0}:\n    if item < best:\n        best = item\nreturn best"], IN_LNE),
       (["mean", "average", "list_average", "average_value"], "the average of the items of the list {0}, which is not empty", ["return sum({0}) / len({0})"], [("[1, 2, 3]",), ("[5]",), ("[2, 4]",), ("[10, 20, 30, 40]",)]),
       (["multiply_all", "list_product", "product_of_all"], "the product of all items of the list {0} (1 for an empty list)", [loop("result = 1", "result *= item", "result")], IN_L),
       (["spread", "max_minus_min", "list_range"], "the largest item of the list {0} minus the smallest item", ["return max({0}) - min({0})"], IN_LNE),
       (["sum_of_absolute_values"], "the sum of the absolute values of the items of the list {0}", ["return sum((abs(item) for item in {0}))"], IN_L),
       (["double_total"], "two times the sum of all items of the list {0}", ["return 2 * sum({0})"], IN_L),
       (["largest_plus_smallest"], "the sum of the largest and the smallest item of the list {0}", ["return max({0}) + min({0})"], IN_LNE)]
for names, doc, bodies, inputs in AGG:
    fam("list_aggregate", names, L1, ["Return %s." % doc], bodies, inputs)
fam("list_aggregate", ["total_length", "sum_of_lengths", "count_all_characters"], W1, ["Return the total number of characters of all strings in the list {0}."], ["return sum((len(s) for s in {0}))", "total = 0\nfor s in {0}:\n    total += len(s)\nreturn total"], IN_W)
fam("list_map", ["lengths", "word_lengths", "length_of_each"], W1, ["Return a list with the length of every string in the list {0}."], ["return [len(s) for s in {0}]"], IN_W)
fam("list_map", ["upper_all", "all_upper_case", "shout_all"], W1, ["Return a list with every string of {0} in upper case."], ["return [s.upper() for s in {0}]"], IN_W)
fam("list_map", ["lower_all", "all_lower_case"], W1, ["Return a list with every string of {0} in lower case."], ["return [s.lower() for s in {0}]"], IN_W)
fam("list_map", ["to_strings", "as_strings"], L1, ["Return a list with every item of {0} turned into a string."], ["return [str(item) for item in {0}]"], IN_L)
fam("list_filter", ["non_empty_strings", "drop_empty", "without_empty_strings"], W1, ["Return a list with the strings of {0} that are not empty."], ["return [s for s in {0} if s != '']", "return [s for s in {0} if len(s) > 0]"], IN_W)
fam("list_filter", ["longer_than", "strings_longer_than"], [("words", "n"), ("strings", "length")], ["Return a list with the strings of {0} that have more than {1} characters."], ["return [s for s in {0} if len(s) > {1}]"], [("['a', 'bb', 'ccc']", "1"), ("[]", "2"), ("['hello', 'hi']", "2"), ("['x', 'yz']", "0")])
for names, test, doc in [(["greater_than", "above", "larger_than", "items_above"], "x > {1}", "greater than {1}"), (["less_than_limit", "below", "items_below"], "x < {1}", "less than {1}"),
                         (["at_least", "items_at_least"], "x >= {1}", "greater than or equal to {1}"), (["not_equal_to", "without_value", "remove_value"], "x != {1}", "not equal to {1}"),
                         (["equal_to", "only_value"], "x == {1}", "equal to {1}")]:
    fam("list_filter", names, [("nums", "limit"), ("numbers", "value"), ("items", "n"), ("values", "k")], ["Return a list with the items of {0} that are %s." % doc], ["return [x for x in {0} if %s]" % test, "result = []\nfor x in {0}:\n    if %s:\n        result.append(x)\nreturn result" % test],
        [("[1, 2, 3]", "2"), ("[]", "5"), ("[5, 1, 9, 5]", "5"), ("[-1, 0, 4]", "0"), ("[7, 7]", "7")])
    fam("list_aggregate", ["count_" + names[0]], [("nums", "limit"), ("numbers", "value"), ("items", "n")], ["Return how many items of {0} are %s." % doc], ["return sum((1 for x in {0} if %s))" % test, "count = 0\nfor x in {0}:\n    if %s:\n        count += 1\nreturn count" % test],
        [("[1, 2, 3]", "2"), ("[]", "5"), ("[5, 1, 9, 5]", "5"), ("[-1, 0, 4]", "0")])
for names, expr, doc in [(["add_to_each", "increase_each", "shift_all"], "x + {1}", "increased by {1}"), (["multiply_each", "scale", "scale_all"], "x * {1}", "multiplied by {1}"), (["subtract_from_each", "decrease_each"], "x - {1}", "decreased by {1}")]:
    fam("list_map", names, [("nums", "k"), ("numbers", "amount"), ("values", "n"), ("items", "value")], ["Return a list with every item of {0} %s." % doc], ["return [%s for x in {0}]" % expr, "result = []\nfor x in {0}:\n    result.append(%s)\nreturn result" % expr],
        [("[1, 2, 3]", "2"), ("[]", "5"), ("[5, 0, -1]", "3"), ("[4]", "10")])
LP = [(["contains", "has_item", "is_member", "includes"], [("items", "target"), ("lst", "value"), ("values", "x")], "Return True if the list {0} contains {1}, otherwise False.", ["return {1} in {0}"], [("[1, 2, 3]", "2"), ("[]", "1"), ("[5, 6]", "7"), ("['a', 'b']", "'a'")]),
      (["does_not_contain", "is_missing", "lacks"], [("items", "target"), ("lst", "value")], "Return True if the list {0} does not contain {1}, otherwise False.", ["return {1} not in {0}"], [("[1, 2, 3]", "2"), ("[]", "1"), ("[5, 6]", "7")]),
      (["is_empty", "has_no_items", "empty"], L1, "Return True if the list {0} has no items, otherwise False.", ["return len({0}) == 0", "return {0} == []"], IN_L),
      (["is_not_empty", "has_items", "non_empty"], L1, "Return True if the list {0} has at least one item, otherwise False.", ["return len({0}) > 0", "return {0} != []"], IN_L),
      (["is_longer_than", "has_more_than"], [("items", "n"), ("lst", "count"), ("values", "size")], "Return True if the list {0} has more than {1} items, otherwise False.", ["return len({0}) > {1}"], [("[1, 2, 3]", "2"), ("[]", "0"), ("[5, 6]", "2"), ("[1]", "0")]),
      (["has_length", "has_exactly"], [("items", "n"), ("lst", "count")], "Return True if the list {0} has exactly {1} items, otherwise False.", ["return len({0}) == {1}"], [("[1, 2, 3]", "3"), ("[]", "0"), ("[5, 6]", "1")]),
      (["starts_with_item", "first_is"], [("items", "target"), ("lst", "value")], "Return True if the first item of the list {0}, which is not empty, is equal to {1}.", ["return {0}[0] == {1}"], [("[1, 2, 3]", "1"), ("[4]", "5"), ("[5, 6]", "6")]),
      (["ends_with_item", "last_is"], [("items", "target"), ("lst", "value")], "Return True if the last item of the list {0}, which is not empty, is equal to {1}.", ["return {0}[-1] == {1}"], [("[1, 2, 3]", "3"), ("[4]", "5"), ("[5, 6]", "5")]),
      (["same_ends", "first_equals_last"], L1, "Return True if the first and the last item of the list {0}, which is not empty, are equal.", ["return {0}[0] == {0}[-1]"], [("[1, 2, 1]",), ("[4]",), ("[5, 6]",), ("[3, 3]",)]),
      (["has_single_item", "is_single"], L1, "Return True if the list {0} has exactly one item, otherwise False.", ["return len({0}) == 1"], IN_L),
      (["sum_is_positive_list", "has_positive_sum"], L1, "Return True if the sum of the items of the list {0} is greater than zero.", ["return sum({0}) > 0"], IN_L)]
for names, args, doc, bodies, inputs in LP:
    fam("list_predicates", names, args, [doc], bodies, inputs)
LI = [(["first", "first_item", "head", "first_element"], "the first item", "{0}[0]"), (["last", "last_item", "final_item", "last_element"], "the last item", "{0}[-1]"),
      (["second", "second_item"], "the second item", "{0}[1]"), (["second_to_last", "item_before_last"], "the item before the last one", "{0}[-2]"),
      (["third_item", "third"], "the third item", "{0}[2]"), (["middle", "middle_item"], "the item in the middle (at position len({0}) // 2)", "{0}[len({0}) // 2]"),
      (["first_plus_last", "sum_of_ends"], "the sum of the first and the last item", "{0}[0] + {0}[-1]"), (["first_times_last"], "the product of the first and the last item", "{0}[0] * {0}[-1]"),
      (["first_minus_last"], "the first item minus the last item", "{0}[0] - {0}[-1]"), (["sum_first_two", "sum_of_first_two"], "the sum of the first two items", "{0}[0] + {0}[1]"),
      (["double_first"], "two times the first item", "{0}[0] * 2"), (["last_minus_first"], "the last item minus the first item", "{0}[-1] - {0}[0]")]
for names, doc, expr in LI:
    fam("list_index", names, L1, ["Return %s of the list {0}, which has at least three items." % doc], ["return " + expr], [("[1, 2, 3]",), ("[4, -1, 0, 7]",), ("[2, 2, 6, 3]",), ("[9, 4, 6, 1, 12]",)])
fam("list_index", ["item_at", "get_item", "nth_item", "element_at"], [("items", "index"), ("lst", "position"), ("values", "k")], ["Return the item of the list {0} at position {1}, counting from 0."], ["return {0}[{1}]"], [("[1, 2, 3]", "0"), ("[4, 5]", "1"), ("[7, 8, 9]", "2")])
LS = [(["first_n", "take", "first_items", "head_items"], [("items", "n"), ("lst", "count"), ("values", "k")], "the first {1} items of the list {0}", "{0}[:{1}]", [("[1, 2, 3]", "2"), ("[4, 5]", "0"), ("[7, 8, 9]", "3"), ("[]", "1")]),
      (["last_n", "last_items", "tail_items"], [("items", "n"), ("lst", "count")], "the last {1} items of the list {0}, where {1} is at least 1", "{0}[-{1}:]", [("[1, 2, 3]", "2"), ("[4, 5]", "1"), ("[7, 8, 9]", "3")]),
      (["drop_n", "skip", "without_first_n"], [("items", "n"), ("lst", "count")], "the list {0} without its first {1} items", "{0}[{1}:]", [("[1, 2, 3]", "2"), ("[4, 5]", "0"), ("[7, 8, 9]", "3")]),
      (["without_first", "rest", "tail", "drop_first"], L1, "the list {0} without its first item", "{0}[1:]", IN_L),
      (["without_last", "drop_last", "all_but_last"], L1, "the list {0} without its last item", "{0}[:-1]", IN_L),
      (["without_ends", "inner_items", "strip_ends"], L1, "the list {0} without its first and last item", "{0}[1:-1]", IN_L),
      (["every_second", "every_other", "even_positions"], L1, "every second item of the list {0}, starting with the first", "{0}[::2]", IN_L),
      (["odd_positions", "skip_one_take_one"], L1, "every second item of the list {0}, starting with the second", "{0}[1::2]", IN_L),
      (["every_third"], L1, "every third item of the list {0}, starting with the first", "{0}[::3]", IN_L),
      (["first_half", "front_half"], L1, "the first half of the list {0} (the first len({0}) // 2 items)", "{0}[:len({0}) // 2]", IN_L),
      (["second_half", "back_half"], L1, "the second half of the list {0} (from position len({0}) // 2)", "{0}[len({0}) // 2:]", IN_L),
      (["first_three", "top_three"], L1, "the first three items of the list {0}", "{0}[:3]", IN_L), (["first_two"], L1, "the first two items of the list {0}", "{0}[:2]", IN_L),
      (["last_two"], L1, "the last two items of the list {0}", "{0}[-2:]", IN_L), (["last_three"], L1, "the last three items of the list {0}", "{0}[-3:]", IN_L),
      (["copy_list", "clone"], L1, "a new list with the same items as the list {0}", "{0}[:]", IN_L)]
for names, args, doc, expr, inputs in LS:
    fam("list_slice", names, args, ["Return %s." % doc], ["return " + expr], inputs)
fam("list_slice", ["between_positions", "slice_from_to", "sublist"], [("items", "start", "end"), ("lst", "a", "b")], ["Return the items of the list {0} from position {1} up to but not including position {2}."], ["return {0}[{1}:{2}]"], [("[1, 2, 3, 4]", "1", "3"), ("[5, 6]", "0", "2"), ("[7, 8, 9]", "2", "2")])
LE = [(["append_item", "add_to_end", "with_appended", "push_back"], "a new list with the items of {0} followed by {1}", "{0} + [{1}]"),
      (["prepend_item", "add_to_front", "with_prepended", "push_front"], "a new list with {1} followed by the items of {0}", "[{1}] + {0}"),
      (["replace_first", "set_first"], "a new list like {0}, which is not empty, with its first item replaced by {1}", "[{1}] + {0}[1:]"),
      (["replace_last", "set_last"], "a new list like {0}, which is not empty, with its last item replaced by {1}", "{0}[:-1] + [{1}]"),
      (["add_to_both_ends", "wrap_with"], "a new list with {1}, then the items of {0}, then {1} again", "[{1}] + {0} + [{1}]"),
      (["append_twice"], "a new list with the items of {0} followed by {1} two times", "{0} + [{1}, {1}]")]
for names, doc, expr in LE:
    fam("list_edit", names, [("items", "value"), ("lst", "item"), ("values", "x"), ("nums", "n")], ["Return %s." % doc], ["return " + expr], [("[1, 2, 3]", "9"), ("[5]", "0"), ("[4, 4]", "4"), ("[7, 8]", "1")])
fam("list_edit", ["insert_at", "insert_item", "put_at"], [("items", "index", "value"), ("lst", "position", "item")], ["Return a new list like {0} with {2} inserted at position {1}."], ["return {0}[:{1}] + [{2}] + {0}[{1}:]"], [("[1, 2, 3]", "1", "9"), ("[]", "0", "5"), ("[4, 5]", "2", "6")])
fam("list_edit", ["remove_at", "delete_at", "without_position"], [("items", "index"), ("lst", "position")], ["Return a new list like {0} without the item at position {1}."], ["return {0}[:{1}] + {0}[{1} + 1:]"], [("[1, 2, 3]", "1"), ("[5]", "0"), ("[4, 5, 6]", "2")])
fam("list_edit", ["join_lists", "concatenate", "combine_lists", "extend_with"], [("first", "second"), ("a", "b"), ("left", "right")], ["Return a new list with the items of {0} followed by the items of {1}."], ["return {0} + {1}"], [("[1, 2]", "[3]"), ("[]", "[]"), ("[5]", "[6, 7]"), ("[]", "[1]")])
fam("list_edit", ["repeat_list", "list_times", "duplicate_list"], [("items", "n"), ("lst", "times"), ("values", "count")], ["Return a new list with the items of {0} repeated {1} times."], ["return {0} * {1}"], [("[1, 2]", "2"), ("[]", "3"), ("[5]", "3"), ("[1, 2]", "0")])
fam("list_edit", ["double_list", "list_twice"], L1, ["Return a new list with the items of {0} followed by the same items again."], ["return {0} + {0}", "return {0} * 2"], IN_L)

# ---- strings
SC = [(["to_upper", "upper_case", "shout", "capitals", "make_upper"], "the string {0} in upper case", "{0}.upper()"), (["to_lower", "lower_case", "whisper", "make_lower"], "the string {0} in lower case", "{0}.lower()"),
      (["text_length", "string_length", "count_characters", "number_of_characters"], "the number of characters in the string {0}", "len({0})"),
      (["double_length"], "two times the number of characters in the string {0}", "len({0}) * 2"), (["length_plus_one"], "the number of characters in the string {0}, plus 1", "len({0}) + 1"),
      (["upper_length"], "the number of characters of the string {0} in upper case", "len({0}.upper())"), (["length_squared"], "the square of the number of characters in the string {0}", "len({0}) ** 2")]
for names, doc, expr in SC:
    fam("string_case_length", names, S1, ["Return %s." % doc], ["return " + expr], IN_S)
fam("string_case_length", ["total_length_of_two", "combined_length", "length_of_both"], [("first", "second"), ("a", "b"), ("s", "t")], ["Return the number of characters of {0} and {1} together."], ["return len({0}) + len({1})", "return len({0} + {1})"], [("'ab'", "'c'"), ("''", "''"), ("'hello'", "'you'"), ("'x'", "''")])
fam("string_case_length", ["is_longer", "longer_than_other"], [("first", "second"), ("a", "b"), ("s", "t")], ["Return True if the string {0} has more characters than the string {1}, otherwise False."], ["return len({0}) > len({1})"], [("'ab'", "'c'"), ("''", "''"), ("'hello'", "'you'"), ("'x'", "'yz'")])
fam("string_case_length", ["same_length", "equal_length"], [("first", "second"), ("a", "b"), ("s", "t")], ["Return True if the strings {0} and {1} have the same number of characters, otherwise False."], ["return len({0}) == len({1})"], [("'ab'", "'cd'"), ("''", "''"), ("'hello'", "'you'"), ("'x'", "'yz'")])
fam("string_case_length", ["length_difference"], [("first", "second"), ("a", "b")], ["Return the number of characters of {0} minus the number of characters of {1}."], ["return len({0}) - len({1})"], [("'ab'", "'c'"), ("''", "''"), ("'hello'", "'you'")])
for k in [1, 2, 3, 4, 5, 8, 10]:
    fam("string_predicates", ["is_longer_than_%d" % k, "has_more_than_%d_characters" % k], S1, ["Return True if the string {0} has more than %d characters, otherwise False." % k], ["return len({0}) > %d" % k], IN_S)
    fam("string_predicates", ["has_length_%d" % k, "has_%d_characters" % k], S1, ["Return True if the string {0} has exactly %d characters, otherwise False." % k], ["return len({0}) == %d" % k], IN_S + [("'%s'" % ("z" * k),)])
SP = [(["starts_with", "begins_with", "has_prefix"], [("s", "prefix"), ("text", "start"), ("word", "beginning")], "the string {0} starts with the string {1}", "{0}.startswith({1})", [("'hello'", "'he'"), ("''", "'a'"), ("'abc'", "'bc'"), ("'x'", "'x'")]),
      (["ends_with", "has_suffix", "finishes_with"], [("s", "suffix"), ("text", "end"), ("word", "ending")], "the string {0} ends with the string {1}", "{0}.endswith({1})", [("'hello'", "'lo'"), ("''", "'a'"), ("'abc'", "'ab'"), ("'x'", "'x'")]),
      (["contains_text", "has_substring", "includes_text"], [("s", "part"), ("text", "piece"), ("word", "sub")], "the string {1} appears inside the string {0}", "{1} in {0}", [("'hello'", "'ell'"), ("''", "'a'"), ("'abc'", "'d'"), ("'x'", "'x'")]),
      (["has_space", "contains_space", "has_blank"], S1, "the string {0} contains a space", "' ' in {0}", IN_S), (["is_empty_string", "is_blank", "has_no_characters"], S1, "the string {0} has no characters", "{0} == ''", IN_S),
      (["is_not_empty_string", "has_characters"], S1, "the string {0} has at least one character", "{0} != ''", IN_S), (["is_all_digits", "only_digits", "is_number_text"], S1, "the string {0} is made of digits only", "{0}.isdigit()", IN_S),
      (["is_all_upper", "is_upper_case", "is_shouting"], S1, "all letters of the string {0} are upper case", "{0}.isupper()", IN_S), (["is_all_lower", "is_lower_case"], S1, "all letters of the string {0} are lower case", "{0}.islower()", IN_S),
      (["is_all_letters", "only_letters"], S1, "the string {0} is made of letters only", "{0}.isalpha()", IN_S), (["starts_with_upper", "is_capitalised", "first_is_upper"], S1, "the first character of the string {0}, which is not empty, is an upper-case letter", "{0}[0].isupper()", IN_SNE),
      (["ends_with_digit", "last_is_digit"], S1, "the last character of the string {0}, which is not empty, is a digit", "{0}[-1].isdigit()", IN_SNE), (["same_first_and_last"], S1, "the first and the last character of the string {0}, which is not empty, are equal", "{0}[0] == {0}[-1]", IN_SNE),
      (["has_dot", "contains_dot"], S1, "the string {0} contains a dot", "'.' in {0}", IN_S + [("'a.b'",)]), (["has_comma"], S1, "the string {0} contains a comma", "',' in {0}", IN_S + [("'a,b'",)]),
      (["is_single_character"], S1, "the string {0} has exactly one character", "len({0}) == 1", IN_S), (["same_text", "strings_equal"], [("first", "second"), ("a", "b")], "the strings {0} and {1} are equal", "{0} == {1}", [("'ab'", "'ab'"), ("''", "''"), ("'a'", "'b'")]),
      (["same_ignoring_case", "equal_ignoring_case"], [("first", "second"), ("a", "b")], "the strings {0} and {1} are equal when upper and lower case are ignored", "{0}.lower() == {1}.lower()", [("'Ab'", "'aB'"), ("''", "''"), ("'a'", "'b'")])]
for names, args, doc, expr, inputs in SP:
    fam("string_predicates", names, args, ["Return True if %s, otherwise False." % doc], ["return " + expr], inputs)
KIND = [("vowels", "ch in 'aeiou'", "are vowels (a, e, i, o, u)"), ("spaces", "ch == ' '", "are spaces"), ("digits", "ch.isdigit()", "are digits"), ("upper_case_letters", "ch.isupper()", "are upper-case letters"),
        ("lower_case_letters", "ch.islower()", "are lower-case letters"), ("letters", "ch.isalpha()", "are letters")]
for letter in "aeostnlr":
    KIND.append(("letter_%s" % letter, "ch == '%s'" % letter, "are the letter %s" % letter))
for kn, ke, kd in KIND:
    fam("string_count", ["count_%s" % kn, "number_of_%s" % kn, "how_many_%s" % kn], S1, ["Return how many characters of the string {0} %s." % kd],
        ["return sum((1 for ch in {0} if %s))" % ke, "count = 0\nfor ch in {0}:\n    if %s:\n        count += 1\nreturn count" % ke], IN_S + [("'banana tree'",), ("'A1 b2 C3'",)])
fam("string_count", ["count_character", "count_char", "occurrences", "count_letter"], [("s", "ch"), ("text", "letter"), ("word", "char")], ["Return how many times the character {1} appears in the string {0}."],
    ["return {0}.count({1})", "count = 0\nfor c in {0}:\n    if c == {1}:\n        count += 1\nreturn count"], [("'banana'", "'a'"), ("''", "'x'"), ("'hello'", "'l'"), ("'abc'", "'z'")])
SI = [(["first_character", "first_char", "first_letter", "initial"], S1, "the first character of the string {0}, which is not empty", "{0}[0]", IN_SNE), (["last_character", "last_char", "last_letter"], S1, "the last character of the string {0}, which is not empty", "{0}[-1]", IN_SNE),
      (["second_character"], S1, "the second character of the string {0}, which has at least two characters", "{0}[1]", [("'hello'",), ("'ab'",), ("'Python'",)]),
      (["without_first_character", "drop_first_char", "rest_of_text"], S1, "the string {0} without its first character", "{0}[1:]", IN_S), (["without_last_character", "drop_last_char", "chop"], S1, "the string {0} without its last character", "{0}[:-1]", IN_S),
      (["without_both_ends", "inner_text", "strip_first_and_last"], S1, "the string {0} without its first and last character", "{0}[1:-1]", IN_S), (["first_two_characters", "first_two_letters"], S1, "the first two characters of the string {0}", "{0}[:2]", IN_S),
      (["first_three_characters", "first_three_letters"], S1, "the first three characters of the string {0}", "{0}[:3]", IN_S), (["last_two_characters", "last_two_letters"], S1, "the last two characters of the string {0}", "{0}[-2:]", IN_S),
      (["last_three_characters"], S1, "the last three characters of the string {0}", "{0}[-3:]", IN_S), (["every_second_character", "every_other_character"], S1, "every second character of the string {0}, starting with the first", "{0}[::2]", IN_S),
      (["first_half_of_text"], S1, "the first half of the string {0} (the first len({0}) // 2 characters)", "{0}[:len({0}) // 2]", IN_S), (["second_half_of_text"], S1, "the second half of the string {0} (from position len({0}) // 2)", "{0}[len({0}) // 2:]", IN_S),
      (["middle_character"], S1, "the character in the middle of the string {0} (at position len({0}) // 2), where {0} is not empty", "{0}[len({0}) // 2]", IN_SNE),
      (["first_and_last_character", "ends_of_text"], S1, "a string made of the first and the last character of the string {0}, which is not empty", "{0}[0] + {0}[-1]", IN_SNE)]
for names, args, doc, expr, inputs in SI:
    fam("string_index_slice", names, args, ["Return %s." % doc], ["return " + expr], inputs)
SN = [("s", "n"), ("text", "count"), ("word", "k")]
for names, doc, expr in [(["first_characters", "take_characters", "prefix_of"], "the first {1} characters of the string {0}", "{0}[:{1}]"), (["skip_characters", "drop_characters"], "the string {0} without its first {1} characters", "{0}[{1}:]"),
                         (["character_at", "char_at", "letter_at"], "the character of the string {0} at position {1}, counting from 0", "{0}[{1}]")]:
    fam("string_index_slice", names, SN, ["Return %s." % doc], ["return " + expr], [("'hello'", "2"), ("'abc'", "0"), ("'Python'", "3"), ("'xy'", "1")])
SB = [(["greet", "greeting", "say_hello", "hello"], [("name",), ("person",)], "the text 'Hello, ' followed by {0}", "'Hello, ' + {0}", [("'Ann'",), ("''",), ("'Bob'",)]),
      (["exclaim", "add_exclamation", "shout_mark"], S1, "the string {0} followed by an exclamation mark", "{0} + '!'", IN_S), (["question", "add_question_mark"], S1, "the string {0} followed by a question mark", "{0} + '?'", IN_S),
      (["in_brackets", "wrap_in_brackets", "parenthesise"], S1, "the string {0} between round brackets", "'(' + {0} + ')'", IN_S), (["in_quotes", "quote"], S1, "the string {0} between double quotes", "'\"' + {0} + '\"'", IN_S),
      (["double_text", "text_twice", "say_twice"], S1, "the string {0} two times in a row", "{0} + {0}", IN_S), (["triple_text", "text_three_times"], S1, "the string {0} three times in a row", "{0} * 3", IN_S),
      (["add_dot", "end_sentence"], S1, "the string {0} followed by a dot", "{0} + '.'", IN_S), (["with_dash_ends"], S1, "the string {0} with a dash before and after it", "'-' + {0} + '-'", IN_S),
      (["upper_with_exclamation", "shout_out"], S1, "the string {0} in upper case followed by an exclamation mark", "{0}.upper() + '!'", IN_S), (["add_ing"], S1, "the string {0} followed by 'ing'", "{0} + 'ing'", IN_S),
      (["add_s", "plural"], S1, "the string {0} followed by the letter s", "{0} + 's'", IN_S), (["mister", "add_mr"], [("name",), ("person",)], "the text 'Mr. ' followed by {0}", "'Mr. ' + {0}", [("'Ann'",), ("''",), ("'Bob'",)])]
for names, args, doc, expr, inputs in SB:
    fam("string_build", names, args, ["Return %s." % doc], ["return " + expr], inputs)
S2 = [("first", "second"), ("a", "b"), ("left", "right"), ("s", "t")]
for names, doc, expr in [(["join_two", "concatenate_strings", "glue", "join_texts"], "the string {0} followed by the string {1}", "{0} + {1}"), (["join_with_space", "full_name", "two_words"], "{0} and {1} joined with one space between them", "{0} + ' ' + {1}"),
                         (["join_with_dash", "hyphenate"], "{0} and {1} joined with a dash between them", "{0} + '-' + {1}"), (["join_with_comma"], "{0} and {1} joined with a comma and a space between them", "{0} + ', ' + {1}"),
                         (["second_then_first", "swap_and_join"], "the string {1} followed by the string {0}", "{1} + {0}"), (["sandwich"], "the string {0}, then the string {1}, then the string {0} again", "{0} + {1} + {0}"),
                         (["join_with_and"], "{0} and {1} joined with the word and between them", "{0} + ' and ' + {1}"), (["join_with_colon"], "{0} and {1} joined with a colon and a space between them", "{0} + ': ' + {1}")]:
    fam("string_build", names, S2, ["Return %s." % doc], ["return " + expr], [("'ab'", "'c'"), ("''", "''"), ("'hello'", "'you'"), ("'x'", "'yz'")])
fam("string_build", ["repeat_text", "repeat_string", "text_times", "multiply_text"], SN, ["Return the string {0} repeated {1} times."], ["return {0} * {1}"], [("'ab'", "2"), ("''", "3"), ("'x'", "5"), ("'hi'", "0")])
fam("string_build", ["number_with_unit", "label_number"], [("n", "unit"), ("value", "name")], ["Return the number {0} as text, followed by a space and the string {1}."], ["return str({0}) + ' ' + {1}"], [("3", "'kg'"), ("0", "'m'"), ("12", "'items'")])
fam("string_build", ["number_as_text", "to_text", "as_string"], P1, ["Return the number {0} as a string."], ["return str({0})"], IN_N)
fam("reverse", ["reverse_text", "backwards", "reversed_string", "flip_text", "mirror_text"], S1, ["Return the string {0} with its characters in reverse order.", "Return the string {0} backwards."], ["return {0}[::-1]"], IN_S)
fam("reverse", ["reversed_items", "backwards_list", "flip_list", "reverse_order", "reverse_items"], L1, ["Return a new list with the items of {0} in reverse order."], ["return {0}[::-1]", "return list(reversed({0}))", "result = []\nfor item in {0}:\n    result = [item] + result\nreturn result"], IN_L)
fam("reverse", ["swap_pair", "swap_two", "reverse_pair", "second_first"], P2, ["Return the two values as a tuple in reverse order: {1} first, then {0}."], ["return ({1}, {0})"], IN_2)
fam("reverse", ["reverse_three", "swap_three", "backwards_triple"], P3, ["Return the three values as a tuple in reverse order: {2}, {1}, {0}."], ["return ({2}, {1}, {0})"], IN_3)
fam("reverse", ["reverse_and_upper"], S1, ["Return the string {0} backwards and in upper case."], ["return {0}[::-1].upper()"], IN_S)
fam("reverse", ["last_to_first_two", "reversed_first_two"], L1, ["Return a list with the second item and then the first item of the list {0}, which has at least two items."], ["return [{0}[1], {0}[0]]"], [("[1, 2, 3]",), ("[4, -1, 0, 7]",), ("[2, 6]",)])
fam("reverse", ["reverse_words_list"], W1, ["Return a new list with the strings of {0} in reverse order."], ["return {0}[::-1]"], IN_W)


def build(n_variants, seed):
    """Instantiate every entry n_variants times; return unique candidate records without tests."""
    rng = random.Random(seed)
    seen, out = set(), []
    for family, names, args, docs, bodies, inputs in E:
        for _ in range(n_variants):
            name, arg, doc, body = rng.choice(names), rng.choice(args), rng.choice(docs), rng.choice(bodies)
            body = body.format(*arg)
            src = "def %s(%s):\n    \"\"\"%s\"\"\"\n%s" % (name, ", ".join(arg), doc.format(*arg), textwrap.indent(body, "    "))
            text = canon(src)
            if text is None or text in seen:
                continue
            seen.add(text)
            calls = ["%s(%s)" % (name, ", ".join(g)) for g in rng.sample(inputs, min(4, len(inputs)))]
            out.append({"text": text, "name": name, "family": family, "calls": calls})
    return out


DRIVER = """
import json
out = []
for src, calls in json.loads(%r):
    ns = {}
    try:
        exec(src, ns)
        out.append([repr(eval(c, ns)) for c in calls])
    except Exception:
        out.append(None)
print("RESULT" + json.dumps(out))
"""


def with_tests(records, batch=150):
    """Run every candidate inside the sandbox to get its outputs; keep those that ran, as assert lines."""
    codes = [DRIVER % json.dumps([[r["text"], r["calls"]] for r in records[g:g + batch]]) for g in range(0, len(records), batch)]
    kept = []
    for g, res in enumerate(run_many(codes, workers=3, timeout_s=20.0)):
        if res.status != "ok" or "RESULT" not in res.stdout:
            continue
        outs = json.loads(res.stdout.split("RESULT", 1)[1])
        for rec, values in zip(records[g * batch:(g + 1) * batch], outs):
            if values is not None:
                rec = dict(rec)
                rec["tests"] = ["assert %s == %s" % (c, v) for c, v in zip(rec.pop("calls"), values)]
                kept.append(rec)
    return kept


def benchmark_keys():
    """Entry-point names and code fingerprints of every benchmark task (HumanEval, MBPP, custom test and practice)."""
    names, keys = set(), set()
    for line in gzip.open("data/bench/HumanEval.jsonl.gz", "rt"):
        t = json.loads(line)
        names.add(t["entry_point"])
        keys.add(alpha_key(t["prompt"] + t["canonical_solution"]))
    for line in open("data/bench/mbpp.jsonl"):
        t = json.loads(line)
        keys.add(alpha_key(t["code"]))
        for test in t["test_list"][:1]:
            names.add(test.split("assert", 1)[1].split("(")[0].strip().lstrip("("))
    for split in ("test", "practice"):
        try:
            for line in open("data/bench/custom_%s.jsonl" % split):
                t = json.loads(line)
                names.add(t["entry_point"])
                keys.add(alpha_key(t["canonical_solution"]))
        except FileNotFoundError:
            print("warning: custom_%s.jsonl is missing, re-run the filter when it exists" % split)
    keys.discard(None)
    return names, keys


def main(out, n_variants, seed):
    names, keys = benchmark_keys()
    cand = build(n_variants, seed)
    by_name = [r for r in cand if r["name"] not in names]
    clean = [r for r in by_name if alpha_key(r["text"]) not in keys]
    kept = with_tests(clean)
    rng = random.Random(seed)
    rng.shuffle(kept)
    n_val = max(200, len(kept) // 50)
    for path, rows in ((out + "/val.jsonl", kept[:n_val]), (out + "/train.jsonl", kept[n_val:])):
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps({"text": r["text"], "name": r["name"], "source": "synthetic", "licence": "own work", "kind": "synthetic", "family": r["family"], "tests": r["tests"]}) + "\n")
    stats = {"entries": len(E), "candidates": len(cand), "removed_benchmark_name": len(cand) - len(by_name), "removed_benchmark_code": len(by_name) - len(clean),
             "failed_in_sandbox": len(clean) - len(kept), "kept": len(kept), "distinct_code": len({alpha_key(r["text"]) for r in kept}), "megabytes": round(sum(len(r["text"]) for r in kept) / 1e6, 2)}
    json.dump(stats, open(out + "/stats.json", "w"), indent=1)
    print(json.dumps(stats))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
