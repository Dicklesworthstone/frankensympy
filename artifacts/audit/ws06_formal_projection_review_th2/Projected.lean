-- Projected statement for profile lean_core_zz_product_v1 (fixed environment: Lean 4 core only).
-- Projection: dense ascending coefficient vectors; equality = coefficient-wise after
-- convolution product, which is structurally recursive and reduces by `decide`.

/-- Pad-and-add: add `scaled` into `acc` indexwise; surplus coefficients appended. -/
def addPad : List Int → List Int → List Int
  | [], r => r
  | s :: stail, [] => s :: stail
  | s :: stail, r0 :: rtail => (s + r0) :: addPad stail rtail

/-- Convolution product of dense ascending ZZ coefficient vectors. -/
def polyMul : List Int → List Int → List Int
  | [], _ => []
  | _, [] => []
  | a :: atail, b => addPad (b.map (· * a)) (0 :: polyMul atail b)

theorem projected_product :
    polyMul [2, -2, 1] [2, 2, 1] = [4, 0, 0, 0, 1] := by decide
#print axioms projected_product
