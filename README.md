# FlowCraft


Around a year ago I purchased a book called "Shaping the Vulcan" by Stephen Liddle, which covers the postwar period of aerodynamic research by the British aircraft industry.<br>
<img src="ShapingTheVulcan.jpg" alt="Book cover" width="100" /><br>

This sparked a fascination as to how wings and fluids interact to generate lift. I decided to write an application to visualise these fluid flows in a more exciting way than graphs on a spreadsheet.

FlowCraft allows you to load any 3D object (in STL format), and then using the Lattice Boltzmann Method (D3Q19 and D3Q15) be able to visualise in realtime (as your machine allows) the fluid flow around that object.<br> The computation can run on your GPU or CPU, but you will need a very powerful machine either way. The GPU backend tends to run faster, but quickly becomes limited by the memory available to the GPU. If your machine is lacking resources, I've included a Panel Method backend (and also a Potential Flow solver, if you really really don't have much computing power), which is less computationally expensive, but isn't accurate...
My relatively old MacStudio Ultra can run the LBM GPU visualisation for 22 million cell lattice at around 1 frame per second.

Due to the large RAM and compute resources required for CFD, I'm not sure how useful this really is as a desktop application but it is fun.

Several visualisations are available; "smoke" trails, colour map (pressure/speed), and voxel space.
<img src="ScreenShot1.jpg" alt="Colour map" width="500" />
<img src="ScreenShot2.jpg" alt="Colour map" width="500" />
<img src="ScreenShot3.jpg" alt="LBM Turbulence" width="500" />

As a nice bonus, the visualiser can also show pressure stress on the airframe, which can help when designing a model to see where extra support might be needed.<br>
<img src="ScreenShot4.jpg" alt="Airframe Stress" width="500" />

## License

FlowCraft is licensed under the [GNU Lesser General Public License v3.0 (LGPL-3.0)](LICENSE).

Third-party libraries and assets retain their respective licenses and copyrights:
- **Dear ImGui** (`FlowCraft/ThirdParty/imgui`) — MIT License.
- **SDL3** — zlib License.
- *Shaping the Vulcan* book cover image (`ShapingTheVulcan.jpg`) — property of its respective copyright holder, shown for commentary/reference.

