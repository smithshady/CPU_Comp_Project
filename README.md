# CPU_Comp_Project
Nvidia Orin AGX GPU Benchmarks

Access:

ssh \<user>@jumpbox.mines.edu

ssh \<user>@isengard.mines.edu

ssh \<user>@hpsslab.com -p 49158 // Orin

- 49158 is the Orin AGX device. It has the VPI library already installed while the Xavier AGX doesn't.

Run:

./subtract.py <cpu||cuda> demo_videos/background_subtractor_input.mp4
