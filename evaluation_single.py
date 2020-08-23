import torch
from torchvision.utils import make_grid
from torchvision.utils import save_image

from util.image import unnormalize


def evaluate(model, dataset, device, output_folder):
  for image, mask, gt, img_name in dataset:

    # https://discuss.pytorch.org/t/converting-numpy-array-to-tensor-on-gpu/19423
    # https://jbencook.com/adding-a-dimension-to-a-tensor-in-pytorch/
    image = image.unsqueeze(0)
    mask = mask.unsqueeze(0)
    # image = torch.stack(image)
    # mask = torch.stack(mask)
    # gt = torch.stack(gt)
    # with torch.no_grad():
    #     output, _ = model(image.to(device), mask.to(device))
    # output = output.to(torch.device('cpu'))
    # output_comp = mask * image + (1 - mask) * output

    print(image.shape)
    print(mask.shape)
    with torch.no_grad():
        output, _ = model(image.to(device), mask.to(device))
    # grid = make_grid(
    #     torch.cat((unnormalize(image), mask, unnormalize(output),
    #                unnormalize(output_comp), unnormalize(gt)), dim=0))
    
    unnormalized_output = unnormalize(output.to(torch.device('cpu')))

    output_path = '{}/{}'.format(output_folder, img_name)
    save_image(unnormalized_output, output_path)



# import torch
# from torchvision.utils import make_grid
# from torchvision.utils import save_image

# from util.image import unnormalize


# def evaluate(model, dataset, device, filename):
#     image, mask, gt, _ = zip(*[dataset[i] for i in range(8)])
#     print(image)
#     print(mask)
#     image = torch.stack(image)
#     mask = torch.stack(mask)
#     print(image.shape)
#     print(mask.shape)
#     gt = torch.stack(gt)
#     with torch.no_grad():
#         output, _ = model(image.to(device), mask.to(device))
#     output = output.to(torch.device('cpu'))
#     output_comp = mask * image + (1 - mask) * output

#     grid = make_grid(
#         torch.cat((unnormalize(image), mask, unnormalize(output),
#                    unnormalize(output_comp), unnormalize(gt)), dim=0))
#     save_image(grid, filename)
