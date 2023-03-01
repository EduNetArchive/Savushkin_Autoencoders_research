import numpy as np
import torch

from sklearn.metrics import accuracy_score, v_measure_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
    
from .models import Autoencoder, TiedAutoencoder


def create_data_loader(root, dataset, transform, batch_size, train=True):
    dataset = dataset(
        root=root,
        train=train,
        download=True,
        transform=transform
    )
    return torch.utils.data.DataLoader(dataset, batch_size, train, num_workers=2)


def create_autoencoder(ae_params, train_loader, fit_params):
    model = Autoencoder(
        ae_params["input_dim"], 
        ae_params["layers_dim"],
        ae_params["latent_dim"],
        ae_params["activation"], 
        ae_params["out_activation"],
        ae_params["device"]
    )
    model = model.to(ae_params["device"])
    model.fit(train_loader, fit_params)
    return model


def create_tied_autoencoder(ae_params, train_loader, fit_params):
    model = TiedAutoencoder(
        ae_params["input_dim"], 
        ae_params["layers_dim"],
        ae_params["latent_dim"],
        ae_params["activation"], 
        ae_params["out_activation"],
        ae_params["alpha"],
        ae_params["device"]
    )
    model = model.to(ae_params["device"])
    model.fit(train_loader, fit_params)
    return model


def compute_metrics(model, dtrain, dtest):
    model.eval()
    x_train = get_samples_numpy(dtrain)
    y_train = dtrain.dataset.targets.numpy()

    x_test = get_samples_numpy(dtest)
    y_test = dtest.dataset.targets.numpy()
    
    metrics = {}
    metrics["eval_train"] = model.evaluate(dtrain)
    metrics["eval_test"] = model.evaluate(dtest)
    
    with torch.no_grad():
        h_train = model.encoder(torch.tensor(x_train, device=model.device)).cpu().numpy()
        h_test = model.encoder(torch.tensor(x_test, device=model.device)).cpu().numpy()
    
    knn = KNeighborsClassifier(n_neighbors=1, algorithm='brute').fit(h_train, y_train)
    kmeans = KMeans(n_clusters=len(np.unique(y_train))).fit(h_train)
    knn_pred = knn.predict(h_test)
    km_pred = kmeans.predict(h_test)
    
    metrics["accuracy"] = accuracy_score(y_test, knn_pred)
    metrics["v_measure"] = v_measure_score(y_test, km_pred)
    return metrics
    
def get_samples_numpy(data_loader):
    return torch.cat(list(
        map(lambda i: data_loader.dataset[i][0], range(len(data_loader.dataset)))
    )).numpy()
    