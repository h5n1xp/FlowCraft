# FlowCraft

<img src="ShapingTheVulcan.jpg" alt="Book cover" width="100" />
Around a year ago I purchased a book called "Shaping the Vulcan" by Stephen Liddle, which covers the postwar period of aerodynamic research by the British aircraft industry. This sparked a fascination as to how wings and fluids interact to generate lift. I decided to write an application to visualise these fluid flows in a more exciting way than graphs on a spreadsheet.

FlowCraft allows you to load any 3D object (in STL format), and then using the Lattice Boltzmann Method (D3Q19 and D3Q15) be able to visualise the fluid flow around that object.<br> The computation can run on your GPU or CPU, but you will need a very powerful machine either way. The GPU backend tends to run faster, but quickly becomes limited by the memory available to the GPU. If your machine is lacking resources, I've included a Panel Method backend (and also a Potential Flow solver, if you really really don't have much computing power), which is less computationally expensive, but isn't accurate...

Due to the large RAM and compute resources required for CFD, I'm not sure how useful this really is as a desktop application but it is fun.

Several visualisations are available; "smoke" trails, colour map, and voxel space.
<img src="ScreenShot1.jpg" alt="Colour map" width="500" />
<img src="ScreenShot2.jpg" alt="Colour map" width="500" />
