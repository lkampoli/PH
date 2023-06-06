#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'OpenFOAMReader'
pH_2D_RANS_kOmegaSSTx_baselinefoam = OpenFOAMReader(FileName='/home/unimelb.edu.au/lcampoli/UoM/Testcases/PeriodicHill/03_Simulation/PH_2D_RANS_kOmegaSSTx_baseline/PH_2D_RANS_kOmegaSSTx_baseline.foam')
pH_2D_RANS_kOmegaSSTx_baselinefoam.MeshRegions = ['internalMesh']
pH_2D_RANS_kOmegaSSTx_baselinefoam.CellArrays = ['Ax', 'Cx', 'Cy', 'Cz', 'DUDt', 'Eta1', 'Eta2', 'Eta3', 'Eta4', 'Eta5', 'I1', 'I2', 'I3', 'I4', 'I5', 'Oijt', 'Q1', 'Q10', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'R', 'Rx', 'Sijt', 'T1', 'T10', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'U', 'aij', 'bij', 'epsilon', 'gradU', 'gradk', 'gradp', 'k', 'nu', 'nut', 'omega', 'p', 'wallDistance', 'wallShearStress', 'yPlus']

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')
# uncomment following to set a specific view size
# renderView1.ViewSize = [1507, 802]

# get color transfer function/color map for 'p'
pLUT = GetColorTransferFunction('p')

# get opacity transfer function/opacity map for 'p'
pPWF = GetOpacityTransferFunction('p')

# show data in view
pH_2D_RANS_kOmegaSSTx_baselinefoamDisplay = Show(pH_2D_RANS_kOmegaSSTx_baselinefoam, renderView1)
# trace defaults for the display properties.
pH_2D_RANS_kOmegaSSTx_baselinefoamDisplay.Representation = 'Surface'
pH_2D_RANS_kOmegaSSTx_baselinefoamDisplay.ColorArrayName = ['POINTS', 'p']
pH_2D_RANS_kOmegaSSTx_baselinefoamDisplay.LookupTable = pLUT
pH_2D_RANS_kOmegaSSTx_baselinefoamDisplay.OSPRayScaleArray = 'p'
pH_2D_RANS_kOmegaSSTx_baselinefoamDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
pH_2D_RANS_kOmegaSSTx_baselinefoamDisplay.SelectOrientationVectors = 'U'
pH_2D_RANS_kOmegaSSTx_baselinefoamDisplay.ScaleFactor = 0.9
pH_2D_RANS_kOmegaSSTx_baselinefoamDisplay.SelectScaleArray = 'p'
pH_2D_RANS_kOmegaSSTx_baselinefoamDisplay.GlyphType = 'Arrow'
pH_2D_RANS_kOmegaSSTx_baselinefoamDisplay.GlyphTableIndexArray = 'p'
pH_2D_RANS_kOmegaSSTx_baselinefoamDisplay.DataAxesGrid = 'GridAxesRepresentation'
pH_2D_RANS_kOmegaSSTx_baselinefoamDisplay.PolarAxes = 'PolarAxesRepresentation'
pH_2D_RANS_kOmegaSSTx_baselinefoamDisplay.ScalarOpacityFunction = pPWF
pH_2D_RANS_kOmegaSSTx_baselinefoamDisplay.ScalarOpacityUnitDistance = 0.2992270212928304

# init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
pH_2D_RANS_kOmegaSSTx_baselinefoamDisplay.OSPRayScaleFunction.Points = [0.0, 0.0, 0.5, 0.0, 3.0, 1.0, 0.5, 0.0]

# reset view to fit data
renderView1.ResetCamera()

# show color bar/color legend
pH_2D_RANS_kOmegaSSTx_baselinefoamDisplay.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# set scalar coloring
ColorBy(pH_2D_RANS_kOmegaSSTx_baselinefoamDisplay, ('POINTS', 'Q1'))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(pLUT, renderView1)

# rescale color and/or opacity maps used to include current data range
pH_2D_RANS_kOmegaSSTx_baselinefoamDisplay.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
pH_2D_RANS_kOmegaSSTx_baselinefoamDisplay.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'Q1'
q1LUT = GetColorTransferFunction('Q1')

# Rescale transfer function
q1LUT.RescaleTransferFunction(0.0105842519552, 0.409889191389)

# get opacity transfer function/opacity map for 'Q1'
q1PWF = GetOpacityTransferFunction('Q1')

# Rescale transfer function
q1PWF.RescaleTransferFunction(0.0105842519552, 0.409889191389)

# current camera placement for renderView1
renderView1.CameraPosition = [4.5, 1.5178571939468384, 9.896009615704545]
renderView1.CameraFocalPoint = [4.5, 1.5178571939468384, -8.456321766609321]
renderView1.CameraParallelScale = 4.7499328837754975

# save screenshot
SaveScreenshot('/home/unimelb.edu.au/lcampoli/UoM/Testcases/PeriodicHill/03_Simulation/PH_2D_RANS_kOmegaSSTx_baseline/Q1.png', renderView1, ImageResolution=[1507, 802])

# set scalar coloring
ColorBy(pH_2D_RANS_kOmegaSSTx_baselinefoamDisplay, ('POINTS', 'Q2'))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(q1LUT, renderView1)

# rescale color and/or opacity maps used to include current data range
pH_2D_RANS_kOmegaSSTx_baselinefoamDisplay.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
pH_2D_RANS_kOmegaSSTx_baselinefoamDisplay.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'Q2'
q2LUT = GetColorTransferFunction('Q2')

# Rescale transfer function
q2LUT.RescaleTransferFunction(0.000119231801364, 0.999734938145)

# get opacity transfer function/opacity map for 'Q2'
q2PWF = GetOpacityTransferFunction('Q2')

# Rescale transfer function
q2PWF.RescaleTransferFunction(0.000119231801364, 0.999734938145)

# current camera placement for renderView1
renderView1.CameraPosition = [4.5, 1.5178571939468384, 9.896009615704545]
renderView1.CameraFocalPoint = [4.5, 1.5178571939468384, -8.456321766609321]
renderView1.CameraParallelScale = 4.7499328837754975

# save screenshot
SaveScreenshot('/home/unimelb.edu.au/lcampoli/UoM/Testcases/PeriodicHill/03_Simulation/PH_2D_RANS_kOmegaSSTx_baseline/Q2.png', renderView1, ImageResolution=[1507, 802])

# set scalar coloring
ColorBy(pH_2D_RANS_kOmegaSSTx_baselinefoamDisplay, ('POINTS', 'Q3'))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(q2LUT, renderView1)

# rescale color and/or opacity maps used to include current data range
pH_2D_RANS_kOmegaSSTx_baselinefoamDisplay.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
pH_2D_RANS_kOmegaSSTx_baselinefoamDisplay.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'Q3'
q3LUT = GetColorTransferFunction('Q3')

# Rescale transfer function
q3LUT.RescaleTransferFunction(1.3473627547e-08, 0.262379705906)

# get opacity transfer function/opacity map for 'Q3'
q3PWF = GetOpacityTransferFunction('Q3')

# Rescale transfer function
q3PWF.RescaleTransferFunction(1.3473627547e-08, 0.262379705906)

# current camera placement for renderView1
renderView1.CameraPosition = [4.5, 1.5178571939468384, 9.896009615704545]
renderView1.CameraFocalPoint = [4.5, 1.5178571939468384, -8.456321766609321]
renderView1.CameraParallelScale = 4.7499328837754975

# save screenshot
SaveScreenshot('/home/unimelb.edu.au/lcampoli/UoM/Testcases/PeriodicHill/03_Simulation/PH_2D_RANS_kOmegaSSTx_baseline/Q3.png', renderView1, ImageResolution=[1507, 802])

# set scalar coloring
ColorBy(pH_2D_RANS_kOmegaSSTx_baselinefoamDisplay, ('POINTS', 'Q4'))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(q3LUT, renderView1)

# rescale color and/or opacity maps used to include current data range
pH_2D_RANS_kOmegaSSTx_baselinefoamDisplay.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
pH_2D_RANS_kOmegaSSTx_baselinefoamDisplay.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'Q4'
q4LUT = GetColorTransferFunction('Q4')

# Rescale transfer function
q4LUT.RescaleTransferFunction(0.0162680502981, 0.999887704849)

# get opacity transfer function/opacity map for 'Q4'
q4PWF = GetOpacityTransferFunction('Q4')

# Rescale transfer function
q4PWF.RescaleTransferFunction(0.0162680502981, 0.999887704849)

# current camera placement for renderView1
renderView1.CameraPosition = [4.5, 1.5178571939468384, 9.896009615704545]
renderView1.CameraFocalPoint = [4.5, 1.5178571939468384, -8.456321766609321]
renderView1.CameraParallelScale = 4.7499328837754975

# save screenshot
SaveScreenshot('/home/unimelb.edu.au/lcampoli/UoM/Testcases/PeriodicHill/03_Simulation/PH_2D_RANS_kOmegaSSTx_baseline/Q4.png', renderView1, ImageResolution=[1507, 802])

# set scalar coloring
ColorBy(pH_2D_RANS_kOmegaSSTx_baselinefoamDisplay, ('POINTS', 'Q5'))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(q4LUT, renderView1)

# rescale color and/or opacity maps used to include current data range
pH_2D_RANS_kOmegaSSTx_baselinefoamDisplay.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
pH_2D_RANS_kOmegaSSTx_baselinefoamDisplay.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'Q5'
q5LUT = GetColorTransferFunction('Q5')

# Rescale transfer function
q5LUT.RescaleTransferFunction(6.44085448585e-05, 0.474435418844)

# get opacity transfer function/opacity map for 'Q5'
q5PWF = GetOpacityTransferFunction('Q5')

# Rescale transfer function
q5PWF.RescaleTransferFunction(6.44085448585e-05, 0.474435418844)

# current camera placement for renderView1
renderView1.CameraPosition = [4.5, 1.5178571939468384, 9.896009615704545]
renderView1.CameraFocalPoint = [4.5, 1.5178571939468384, -8.456321766609321]
renderView1.CameraParallelScale = 4.7499328837754975

# save screenshot
SaveScreenshot('/home/unimelb.edu.au/lcampoli/UoM/Testcases/PeriodicHill/03_Simulation/PH_2D_RANS_kOmegaSSTx_baseline/Q5.png', renderView1, ImageResolution=[1507, 802])

# set scalar coloring
ColorBy(pH_2D_RANS_kOmegaSSTx_baselinefoamDisplay, ('POINTS', 'Q6'))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(q5LUT, renderView1)

# rescale color and/or opacity maps used to include current data range
pH_2D_RANS_kOmegaSSTx_baselinefoamDisplay.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
pH_2D_RANS_kOmegaSSTx_baselinefoamDisplay.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'Q6'
q6LUT = GetColorTransferFunction('Q6')

# Rescale transfer function
q6LUT.RescaleTransferFunction(2.07364334037e-12, 0.718976974487)

# get opacity transfer function/opacity map for 'Q6'
q6PWF = GetOpacityTransferFunction('Q6')

# Rescale transfer function
q6PWF.RescaleTransferFunction(2.07364334037e-12, 0.718976974487)

# current camera placement for renderView1
renderView1.CameraPosition = [4.5, 1.5178571939468384, 9.896009615704545]
renderView1.CameraFocalPoint = [4.5, 1.5178571939468384, -8.456321766609321]
renderView1.CameraParallelScale = 4.7499328837754975

# save screenshot
SaveScreenshot('/home/unimelb.edu.au/lcampoli/UoM/Testcases/PeriodicHill/03_Simulation/PH_2D_RANS_kOmegaSSTx_baseline/Q6.png', renderView1, ImageResolution=[1507, 802])

# set scalar coloring
ColorBy(pH_2D_RANS_kOmegaSSTx_baselinefoamDisplay, ('POINTS', 'Q7'))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(q6LUT, renderView1)

# rescale color and/or opacity maps used to include current data range
pH_2D_RANS_kOmegaSSTx_baselinefoamDisplay.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
pH_2D_RANS_kOmegaSSTx_baselinefoamDisplay.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'Q7'
q7LUT = GetColorTransferFunction('Q7')

# Rescale transfer function
q7LUT.RescaleTransferFunction(0.000199027985218, 0.996649861336)

# get opacity transfer function/opacity map for 'Q7'
q7PWF = GetOpacityTransferFunction('Q7')

# Rescale transfer function
q7PWF.RescaleTransferFunction(0.000199027985218, 0.996649861336)

# current camera placement for renderView1
renderView1.CameraPosition = [4.5, 1.5178571939468384, 9.896009615704545]
renderView1.CameraFocalPoint = [4.5, 1.5178571939468384, -8.456321766609321]
renderView1.CameraParallelScale = 4.7499328837754975

# save screenshot
SaveScreenshot('/home/unimelb.edu.au/lcampoli/UoM/Testcases/PeriodicHill/03_Simulation/PH_2D_RANS_kOmegaSSTx_baseline/Q7.png', renderView1, ImageResolution=[1507, 802])

# set scalar coloring
ColorBy(pH_2D_RANS_kOmegaSSTx_baselinefoamDisplay, ('POINTS', 'Q8'))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(q7LUT, renderView1)

# rescale color and/or opacity maps used to include current data range
pH_2D_RANS_kOmegaSSTx_baselinefoamDisplay.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
pH_2D_RANS_kOmegaSSTx_baselinefoamDisplay.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'Q8'
q8LUT = GetColorTransferFunction('Q8')

# Rescale transfer function
q8LUT.RescaleTransferFunction(6.67053023928e-10, 0.973178744316)

# get opacity transfer function/opacity map for 'Q8'
q8PWF = GetOpacityTransferFunction('Q8')

# Rescale transfer function
q8PWF.RescaleTransferFunction(6.67053023928e-10, 0.973178744316)

# current camera placement for renderView1
renderView1.CameraPosition = [4.5, 1.5178571939468384, 9.896009615704545]
renderView1.CameraFocalPoint = [4.5, 1.5178571939468384, -8.456321766609321]
renderView1.CameraParallelScale = 4.7499328837754975

# save screenshot
SaveScreenshot('/home/unimelb.edu.au/lcampoli/UoM/Testcases/PeriodicHill/03_Simulation/PH_2D_RANS_kOmegaSSTx_baseline/Q8.png', renderView1, ImageResolution=[1507, 802])

# set scalar coloring
ColorBy(pH_2D_RANS_kOmegaSSTx_baselinefoamDisplay, ('POINTS', 'Q9'))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(q8LUT, renderView1)

# rescale color and/or opacity maps used to include current data range
pH_2D_RANS_kOmegaSSTx_baselinefoamDisplay.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
pH_2D_RANS_kOmegaSSTx_baselinefoamDisplay.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'Q9'
q9LUT = GetColorTransferFunction('Q9')

# current camera placement for renderView1
renderView1.CameraPosition = [4.5, 1.5178571939468384, 9.896009615704545]
renderView1.CameraFocalPoint = [4.5, 1.5178571939468384, -8.456321766609321]
renderView1.CameraParallelScale = 4.7499328837754975

# save screenshot
SaveScreenshot('/home/unimelb.edu.au/lcampoli/UoM/Testcases/PeriodicHill/03_Simulation/PH_2D_RANS_kOmegaSSTx_baseline/Q9.png', renderView1, ImageResolution=[1507, 802])

# set scalar coloring
ColorBy(pH_2D_RANS_kOmegaSSTx_baselinefoamDisplay, ('POINTS', 'Q10'))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(q9LUT, renderView1)

# rescale color and/or opacity maps used to include current data range
pH_2D_RANS_kOmegaSSTx_baselinefoamDisplay.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
pH_2D_RANS_kOmegaSSTx_baselinefoamDisplay.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'Q10'
q10LUT = GetColorTransferFunction('Q10')

# Rescale transfer function
q10LUT.RescaleTransferFunction(0.51261484623, 0.560324192047)

# get opacity transfer function/opacity map for 'Q10'
q10PWF = GetOpacityTransferFunction('Q10')

# Rescale transfer function
q10PWF.RescaleTransferFunction(0.51261484623, 0.560324192047)

# current camera placement for renderView1
renderView1.CameraPosition = [4.5, 1.5178571939468384, 9.896009615704545]
renderView1.CameraFocalPoint = [4.5, 1.5178571939468384, -8.456321766609321]
renderView1.CameraParallelScale = 4.7499328837754975

# save screenshot
SaveScreenshot('/home/unimelb.edu.au/lcampoli/UoM/Testcases/PeriodicHill/03_Simulation/PH_2D_RANS_kOmegaSSTx_baseline/Q10.png', renderView1, ImageResolution=[1507, 802])

#### saving camera placements for all active views

# current camera placement for renderView1
renderView1.CameraPosition = [4.5, 1.5178571939468384, 9.896009615704545]
renderView1.CameraFocalPoint = [4.5, 1.5178571939468384, -8.456321766609321]
renderView1.CameraParallelScale = 4.7499328837754975

#### uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).