import astra
import torch
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
from torchvision import transforms

class CT():
    def __init__(self, noise_model=None, type=1) -> None:
        self.angles = np.linspace(0, np.pi, 180, endpoint=False) 
        self.noise_model = noise_model if noise_model != None else {'noise_type':'g', 'sigma':0.1} # Gaussian
        self.name = 'ct'
        self.type = type

    def A(self, x, add_noise=False, angles=None):
        assert angles is not None
        self.angles = angles

        if type(x) is not np.ndarray:
            x = x.numpy()

        length = int((x.shape[1]**2 + x.shape[2]**2)**0.5 + 0.5)
        # length=10
        tensor = np.zeros((x.shape[0], len(self.angles), length))
        for i in range(x.shape[0]):
            # Specify projection geometry
            proj_geom = astra.create_proj_geom('parallel', 1, length, self.angles)

            # Specify volume geometry
            vol_geom = astra.create_vol_geom(x.shape[1], x.shape[2])

            # Create a projector object
            proj_id = astra.create_projector('cuda', proj_geom, vol_geom)

            # Create a sinogram
            sinogram_id, tensor[i] = astra.create_sino(x[i], proj_id)


            # # Motion artifact
            # motion_shift = 10  # Number of pixels to shift the sinogram
            # tensor[i] = np.roll(tensor[i], motion_shift, axis=0)

            # # ring artifact (additive sinusoidal pattern)
            # ring_amplitude = 100  # Amplitude of the sinusoidal pattern
            # ring_period = 2*np.pi  # Period of the sinusoidal pattern (in pixels)
            # ring_artifact = ring_amplitude * np.sin(np.linspace(0, 2*np.pi, tensor[i].shape[0]) * (2*np.pi / ring_period))
            # tensor[i] = tensor[i] + ring_artifact.reshape((-1, 1))

            # Metal
            if self.type == 4:
                metal_angle0 = [20, 40, 60, 80, 100,]  # Angles where metal artifact occurs
                # angles = range(num_angles)
                # metal_angles=np.random.choice(angles, size=60, replace=False) sparse
                for angle0 in metal_angle0:
                    metal_angles = range(angle0, angle0+10, 1)
                    reduction = 200.0  # Intensity of metal artifact
                    for angle in metal_angles:
                        tensor[i][angle, :] += reduction

            tensor[i] = tensor[i]
        tensor = torch.from_numpy(tensor)
        
        if add_noise:
            tensor = self.add_noise(tensor)

        return tensor, proj_id
    
    def A_dagger(self, x, proj_id=None, algorithm='BP'):
        assert proj_id != None and type(proj_id) == int
        
        if type(x) is not np.ndarray:
            x = x.numpy()

        # get projection and volume geometry of the FP (the FP is defined above in this case) 
        proj_geom = astra.projector.projection_geometry(proj_id)
        vol_geom = astra.projector.volume_geometry(proj_id)
        tensor = np.zeros((x.shape[0], vol_geom['GridRowCount'], vol_geom['GridColCount']))

        for i in range(x.shape[0]):
            # id of the reoncstructed data
            recon_id = astra.data2d.create('-vol', vol_geom)
            
            # run reconstruction algorithm
            cfg = astra.astra_dict(algorithm + '_CUDA')
            cfg['ProjectionDataId'] = astra.data2d.create('-sino', proj_geom, x[i])
            cfg['ReconstructionDataId'] = recon_id
            fbp_id = astra.algorithm.create(cfg)
            astra.algorithm.run(fbp_id)
            astra.algorithm.delete(fbp_id)
            tensor[i] = astra.data2d.get(recon_id)

        return torch.from_numpy(tensor)

    def add_noise(self, x):
        if self.noise_model['sigma'] > 0:
            noise = torch.randn_like(x) * self.noise_model['sigma']
            x = x + noise
        return x
    
    def get_angles(self):
        # Sparse-view
        if self.type == 1:
            num_angles = random.randint(5, 180)
            return np.linspace(0, np.pi, 60, endpoint=False)

        # Limited-angle
        if self.type == 2:
            lb = 20
            ub = 90
            angles = np.linspace((lb/180)*np.pi, (ub/180)*np.pi, num_angles, endpoint=False)
            return np.random.choice(angles, size=20, replace=False)

        # Motion
        if self.type == 3 or self.type == 4:
            return np.linspace(0, np.pi, num_angles, endpoint=False)

if __name__ == "__main__":
    image = Image.open('D:\EI-Draft\operators/radon\chest_phantom_resized.png')
    if image.mode != 'RGB':
        image = image.convert('RGB')

    preprocess = transforms.Compose([
        transforms.CenterCrop((256,256)),
        transforms.ToTensor(),
        transforms.Grayscale()
    ])
    data = preprocess(image).numpy()
    ct = CT(type=4) # random angles=np.random.choice(angles, size=20, replace=False)
    sinogram, proj_id = ct.A(x=data, add_noise=True)
    bp_reconstruction = ct.A_dagger(x=sinogram, proj_id=proj_id)
    fbp_reconstruction = ct.A_dagger(x=sinogram, proj_id=proj_id, algorithm='FBP')
    
    fig, (ax1,ax2,ax3,ax4) = plt.subplots(1,4)
    ax1.imshow(data.squeeze(0), cmap='grey')
    ax1.set_title('Gound truth')

    ax2.imshow(sinogram.squeeze(0), cmap='grey')
    ax2.set_title(f'Sinogram (10 random angles \n between 20' + r'$\degree$' +  'and 90' + r'$\degree$)')

    ax3.imshow(bp_reconstruction.squeeze(0), cmap='grey')
    ax3.set_title('Reonctruction via BP')

    ax4.imshow(fbp_reconstruction.squeeze(0), cmap='grey')
    ax4.set_title('Reonctruction via FBP')

    plt.show()