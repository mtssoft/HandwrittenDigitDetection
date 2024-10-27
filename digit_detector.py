import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as  np
import random

# Veriyi normalize etmek için transform işlemi.
# Birden fazla ön işleme adımını sırayla uygulamak için kullanılır.
# Yani veri setini modelin anlayabileceği formata dönüştürmek ve veriyi daha iyi öğrenebilmesi için ön işlem yapmak amacıyla kullanılır.
# transforms.Compose fonksiyonu içerisinde iki işlem uygulanıyor.
# transforms.ToTensor() sayısal matrisler olarak tutulan MNIST veri setini PyTorch'un tensör veri tipine dönüştürür.
# transforms.Normalize((0.5,), (0.5,)) ise veriyi normalize eder. Yani verinin her piksel değerini -1 ile 1 arasına sıkıştırır.
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# MNIST veri setini yükleme (eğitim ve test)
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Veriyi DataLoader'a yükleme (batch size = 64)
# Veri setini modelde eğitmek ve test ederken küçük parçalar halinde ve karışık bir şekilde yükleyiciler tanımlanmıştır.
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# HandWrittenRecogDNN DNN modeli tanımlaması
class HandWrittenDigitRecogDNN(nn.Module):
    def __init__(self):
        super(HandWrittenDigitRecogDNN, self).__init__()
        # 784 nöronlu giriş katmanı ve 128 nöronlu 1. gizli katman tanımlaması
        self.fc1 = nn.Linear(28*28, 128)
        # 128 nöronlu 2. gizli katman tanımlaması
        self.fc2 = nn.Linear(128, 128)
        # 10 Nöronlu çıkış katmanı tanımlaması
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        # Girdi katmanında 28x28 görüntüyü tek boyutlu hale getilir
        x = x.view(-1, 28*28)
        # 1.gizli katman ReLu aktivasyon fonksiyonunu uygulanır
        x = F.relu(self.fc1(x))
        # 2 gizli katman 1.gizli katmandan alınan çıktıları alır ve ReLU aktivasyonunu uygulanır
        x = F.relu(self.fc2(x))
        # Çıkış katmanında 2.gizli katmandan alınan çıktılar alınır log-softmax aktivasyon fonksiyonu uygulanır ve modelin çıktısı
        # elde edilir
        x = F.log_softmax(self.fc3(x), dim=1)
        return x

# Modeli ve optimizasyonu ayarlama
# Model objemizden bir instance yaratılır
model = HandWrittenDigitRecogDNN()
# torch.optim kütüphanesi kullanılarak Adam algoritmasını kullanan optimizasyon tanımlanır.
# bu tanımlamadaki torch.optim kütüphanesinden gelen Adam fonksiyonuna 2 parametre verilir.
# 1.parametre DNN objemizin parametreleridir ve bunu almayı sağlayan fonksiyon torch.nn kütüphanesinden gelir
# lr ise öğrenme oranı (Learning Rate) parametresidir. Her güncelleme adımında ağırlıkların ne kadar değişeceğini belirtir.
optimizer = optim.Adam(model.parameters(), lr=0.001)
#Kayıp hesaplama olarak da torch.nn'den gelen CrossEntropyLoss fonksiyonu tanımlanır
criterion = nn.CrossEntropyLoss()
# Eğitim Fonksiyonu
def train(model, device, train_loader, optimizer, epoch):
    #model eğitim moduna alınır. torch.nn kütüphanesinden gelen özelliktir
    model.train()
    # başlangıç kayıp ve doğru tahmin değerleri atanır
    total_loss = 0
    correct = 0
    # eğitim verisini batch'ler halinde döngü ile parça parça işleyelim
    for batch_idx, (data, target) in enumerate(train_loader):
        #veri seçilen cihaza (GPU ya da CPU'ya) yönlendirilir
        data, target = data.to(device), target.to(device)
        #önceki batch'e ait gradyanları sıfırlanır
        optimizer.zero_grad()
        #modelin tahmin ettiği sonuç alınır. Model objesinde tanımladığımız forward fonksiyonu burada çalışır
        output = model(data)
        #kayıp hesaplanır
        loss = criterion(output, target)
        # geri yayılım ile gradyanlar hesaplanır
        loss.backward()
        # hesaplanan gradyanlar kullanılara modelin ağırlıkları güncellenir
        optimizer.step()

        # toplam hata ve doğru tahmin değerleri hesaplanır
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    #dönemdeki (epoch) doğruluk ve hata değerleri hesaplanır
    accuracy = correct / len(train_loader.dataset)
    avg_loss = total_loss / len(train_loader)
    # Hangi dönemin (epoch) eğitiminin yapıldığı, toplam hata ve doğuluk oranları yazdırılır.
    print(f'Dönem {epoch}: Kayıp = {avg_loss:.4f}, Doğruluk = {accuracy:.4f}')
    # eğitim döneminin doğruluk ve hata hesaplamaları döndürülür
    return avg_loss, accuracy

# test fonksiyonu
def test(model, device, test_loader):
    # modeli hesaplama moduna alınır
    model.eval()
    # başlangıç kayıp ve doğru tahmin tanımlamaları
    test_loss = 0
    correct = 0
    # gradyan hesaplaması devre dışı bırakılır
    with torch.no_grad():
        # test veri seti yüklenir ve döngü ile hepsi için model tahmini üretilir
        for data, target in test_loader:
            # veri cihaza (GPU veya CPU'ya yönlendirilir
            data, target = data.to(device), target.to(device)
            # model tahmini üretilir
            output = model(data)
            # test verisinin hata miktarı ve doğru tahmin hesaplanır
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    #test veri setinde yapulan testin doğruluk ve hata değerleri hesaplanır
    test_loss /= len(test_loader)
    accuracy = correct / len(test_loader.dataset)
    #hesaplanan doğruluk ve hata değerleri ekrana yazdırılır
    print(f'Test: Kayıp = {test_loss:.4f}, Doğruluk = {accuracy:.4f}')
    # test sonuçları döndürülür
    return test_loss, accuracy

# Eğitim ve test döngüsü
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
epochs = 5
train_losses, train_accuracies = [], []
test_losses, test_accuracies = [], []

for epoch in range(1, epochs + 1):
    train_loss, train_acc = train(model, device, train_loader, optimizer, epoch)
    test_loss, test_acc = test(model, device, test_loader)

    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)

# Loggsoftmax aktivasyon fonksiyonu grafiği çizdiren fonksiyon
def plot_logsoftmax():
    # Giriş değerleri oluşturma (-10 ile 10 arasında 100 değer)
    x_values = np.linspace(-10, 10, 100)

    # LogSoftmax fonksiyonunu uygula
    x_tensor = torch.tensor(x_values, dtype=torch.float32).unsqueeze(0)  # 2 boyutlu hale getir
    y_values = F.log_softmax(x_tensor, dim=1).squeeze().detach().numpy()

    # Grafiği çizdir
    plt.figure(figsize=(10, 5))
    plt.plot(x_values, y_values, label='LogSoftmax', color='blue')
    plt.xlabel("Giriş Değerleri")
    plt.ylabel("LogSoftmax Çıkış Değerleri")
    plt.title("LogSoftmax Aktivasyon Fonksiyonu Grafiği")
    plt.axhline(0, color='black', lw=0.5, ls='--')  # Y ekseninde sıfır çizgisi
    plt.axvline(0, color='black', lw=0.5, ls='--')  # X ekseninde sıfır çizgisi
    plt.legend()
    plt.grid(True)
    plt.show()

# LogSoftmax grafiğini çizme fonksiyonunu çağır
plot_logsoftmax()

# Eğitim ve test sonuçlarını görselleştirme
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Eğitim Kayıp')
plt.plot(test_losses, label='Test Kayıp')
plt.title('Dönemler Süresince Kayıp')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Eğitim Doğruluk')
plt.plot(test_accuracies, label='Test Doğruluk')
plt.title('Dönemler Süresince Doğruluk')
plt.legend()

plt.show()

# Test veri setinden rastgele bir örnek seçelim
def visualize_random_test_sample(model, device, test_loader):
    # Test veri setinden rastgele bir örnek alınır
    # Örneğin 0. indisli görseli seçebilirsiniz veya random bir örnek almak için
    # index = np.random.randint(len(test_dataset)) kullanılabilir
    index = np.random.randint(len(test_dataset))
    image, label = test_dataset[index]

    # Modeli değerlendirme moduna al
    model.eval()

    # Girişi uygun forma getir ve cihazına gönder (CPU veya GPU)
    image = image.to(device).unsqueeze(0)  # Modelin beklediği boyuta uyacak şekilde yeniden boyutlandırıyoruz

    # Modeli kullanarak tahmin yap
    with torch.no_grad():
        output = model(image)
        predicted_label = output.argmax(dim=1, keepdim=True).item()

    # Sonuçları yazdır
    print(f"Gerçek etiket: {label}")
    print(f"Tahmin edilen etiket: {predicted_label}")

    # Görseli göster
    plt.imshow(image.cpu().squeeze(), cmap='gray')
    plt.title(f"Gerçek: {label}, Tahmin: {predicted_label}")
    plt.axis('off')
    plt.show()
# Test veri setinden rastgele bir örneği tahmin edip görselleştirelim
visualize_random_test_sample(model, device, test_loader)
