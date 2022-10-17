batch_size = 128
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

testloader = torch.utils.data.DataLoader(datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([transforms.ToTensor(),normalize,])), batch_size=batch_size, shuffle=False)

model_new = torch.load('D:torch_exp_vgg_2021-11-03_1.pth', map_location=torch.device('cpu'))
model_new.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model_new(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')