'''
functions and templates that are often needed
helpful function

How find smth?
ctrl + f -> name from list below(without <>)

This file contains:
 - <Dataset template> image dataset template
 - <image show> image show function(helpful for img generators)
 - <Segmentation augumentation>Custom Transform functions for Segmentation
 - <save load> model saving and loading


'''


### Dataset template ###
################################################################################################################
class LaserDS(Dataset):
    def __init__(self, source_path, seg_path, image_size = 256, transform = None):
        
        self.source_path = source_path
        self.source_file_names = [file for file in os.listdir(source_path) if not file.endswith('.npz')]

        self.image_size = image_size


        self.len = len(self.source_file_names)

        
        if transform is None:
            transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5],), 
                            transforms.Resize(self.image_size, antialias=True),
                        ])
        self.transform = transform

        self.my_trans = [r_rotation, r_hflip, r_vflip]

    def __len__(self):
        return self.len
    
    def __getitem__(self, indx):
        img_name = self.source_file_names[indx]

        img_full_path =  os.path.join(self.source_path, img_name)

        img_t = self.transform(Image.open(img_full_path))

        return img_t
  




### image show ###
################################################################################################################
def show_real_and_fake(gen_W, gen_S, dataset, epo=0, save_not_show=True, n_imgs=7):

    std = mean = 0.5
    gen_W.eval()
    gen_S.eval()
    plt.figure(figsize=(20, 20))
    
    for i in range(n_imgs):

        real_summ, real_win = dataset[i]
        with torch.no_grad():
            fake_summ = gen_S(torch.unsqueeze(real_win.to(config.DEVICE, dtype=torch.float), 0))
            fake_win = gen_W(torch.unsqueeze(real_summ.to(config.DEVICE, dtype=torch.float), 0))
        fake_summ = fake_summ[0].cpu()
        fake_win = fake_win[0].cpu()


        real_summ = real_summ * std + mean
        fake_summ = fake_summ * std + mean

        real_win = real_win * std + mean
        fake_win = fake_win * std + mean
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plt.subplot(4, n_imgs, i+1)
            plt.axis("off")
            plt.imshow(real_summ.permute(1, 2, 0))
            plt.title('Real Summer')

            plt.subplot(4, n_imgs, i+n_imgs+1)
            plt.axis("off")
            plt.imshow(fake_win.permute(1, 2, 0))
            plt.title('Fake winter')

            plt.subplot(4, n_imgs, i+n_imgs*2+1)
            plt.axis("off")
            plt.imshow(real_win.permute(1, 2, 0))
            plt.title('Real Winter')

            plt.subplot(4, n_imgs, i+n_imgs*3+1)
            plt.axis("off")
            plt.imshow(fake_summ.permute(1, 2, 0))
            plt.title('Fake Summer')

### Segmentation augumentation ###
################################################################################################################
class RandomRotationWithMask(transforms.RandomRotation):
    def __call__(self, img, mask):
        angle = self.get_params(self.degrees)
        return transforms.functional.rotate(img, angle), transforms.functional.rotate(mask, angle)
    
class RandomHFlipWithMask:
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, img, mask):
        if random.random() < self.p:
            return transforms.functional.hflip(img), transforms.functional.hflip(mask)
        return img, mask
    
class RandomVFlipWithMask:
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, img, mask):
        if random.random() < self.p:
            return transforms.functional.vflip(img), transforms.functional.vflip(mask)
        return img, mask


class RandomBlurWithMask:
    def __init__(self, blur_radius = 2):
        self.blur_radius = blur_radius
    def __call__(self, img, mask):
        return transforms.functional.gaussian_blur(img, self.blur_radius), mask


class RandomNoiseWithMask:
    def __init__(self, p=0.2):
        self.p = p
    def __call__(self, img, mask):
        noise = torch.randn_like(img)*self.p
        return img+noise, mask

def segmentation_augumentation_example():
    image_path = "laser/img_1.jpeg"
    mask_path = "laser_seg/img_1.jpg"

    tot = transforms.ToTensor()

    image = tot(Image.open(image_path))
    mask = tot(Image.open(mask_path))

    r_rotation = RandomRotationWithMask(degrees=45)
    random_flip = RandomHFlipWithMask(p=1)
    #random_shear = RandomShearWithMask(degrees=(-30, 30))
    random_blur = RandomBlurWithMask(blur_radius=41)
    random_noise = RandomNoiseWithMask(p=0.5)



    for transf, name in zip([r_rotation, random_flip, random_blur, random_noise], ['rotation', 'r_flip', 'r_blur', 'r_noise']):
        transformed_image, transformed_mask = transf(image, mask)
        fig, axs = plt.subplots(2, 2, figsize=(5, 5))
        axs[0, 0].imshow(image.permute(1, 2, 0))
        axs[0, 0].set_title('Original Image')
        axs[0, 0].axis('off')

        axs[0, 1].imshow(mask.permute(1, 2, 0), cmap='gray')
        axs[0, 1].set_title('Original Mask')
        axs[0, 1].axis('off')

        axs[1, 0].imshow(transformed_image.permute(1, 2, 0))
        axs[1, 0].set_title(f'{name} Image')
        axs[1, 0].axis('off')

        axs[1, 1].imshow(transformed_mask.permute(1, 2, 0), cmap='gray')
        axs[1, 1].set_title(f'{name} Mask')
        axs[1, 1].axis('off')

        plt.show()

### save load ###
################################################################################################################
# The functions save_model and load_checkpoint was taken from: 
# https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/GANs/Pix2Pix/utils.py

def save_checkpoint(model, optimizer, filename):
    
    states = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(states, filename)

def load_checkpoint(checkpoint_file, model, optimizer, lr):
    
    checkpoint = torch.load(checkpoint_file, map_location=DEVICE)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

# example - saving and loading cyclegan
def save_cycle_gan(gen_W, gen_S, opt_gen, disc_W, disc_S, opt_disc, time):

    # path looks like .../IMAGE_SIZE/2023_01_01_01h_01m/filename

    ###I was thinking of adding more information about LAMBDA_CYCLE, but decided dont do it
    #current_confi = f'imgs_{IMAGE_SIZE}_lamb_{LAMBDA_CYCLE}'

    time = time.strftime('%Y_%m_%d_%Hh%Mm')
    path = os.path.join(SAVED_MODELS_DIR, str(IMAGE_SIZE))
    path = os.path.join(path, time)

    if not os.path.exists(path):
        os.makedirs(path)
    
    print('saving cyclegan...')
    save_checkpoint(gen_W, opt_gen, filename=os.path.join(path, 'gen_W'))
    save_checkpoint(gen_S, opt_gen, filename=os.path.join(path, 'gen_S'))
    save_checkpoint(disc_W, opt_disc, filename=os.path.join(path, 'disc_W'))
    save_checkpoint(disc_S, opt_disc, filename=os.path.join(path, 'disc_S'))
    print('saved')

def load_cycle_gan_inplace(gen_W, gen_S, opt_gen, disc_W, disc_S, opt_disc, dir_path):
    print('loading model', dir_path)
    load_checkpoint(
        os.path.join(dir_path, 'gen_W'),
        gen_W,
        opt_gen,
        LEARNING_RATE,
    )
    load_checkpoint(
        os.path.join(dir_path, 'gen_S'),
        gen_S,
        opt_gen,
        LEARNING_RATE,
    )
    load_checkpoint(
        os.path.join(dir_path, 'disc_W'),
        disc_W,
        opt_disc,
        LEARNING_RATE,
    )
    load_checkpoint(
        os.path.join(dir_path, 'disc_S'),
        disc_S,
        opt_disc,
        LEARNING_RATE,
    )
    print('model loaded')


