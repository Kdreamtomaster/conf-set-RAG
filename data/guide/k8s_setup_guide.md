# 🛠 Kubernetes 환경 설정 가이드

이 문서는 **Kubernetes(K8s) 환경 설정** 및 배포를 위한 가이드입니다. 기본적인 설치부터, 클러스터 설정, 네트워크 구성, 애플리케이션 배포까지 포함합니다.

---

## 1️⃣ Kubernetes 환경 개요

### ✅ 지원 환경
- 운영체제: Ubuntu 20.04 / Debian 11 / CentOS 8 / macOS (Apple Silicon 지원)
- Kubernetes 버전: 1.26+
- 컨테이너 런타임: `containerd` 또는 `Docker`

---

## 2️⃣ 필수 패키지 설치

### 🔹 Linux (Ubuntu / Debian 계열)
```sh
sudo apt update && sudo apt install -y apt-transport-https ca-certificates curl
sudo curl -fsSL https://pkgs.k8s.io/core:/stable:/v1.26/deb/Release.key | sudo tee /etc/apt/trusted.gpg.d/kubernetes.asc
sudo apt-add-repository "deb https://pkgs.k8s.io/core:/stable:/v1.26/deb/ /"
sudo apt update
sudo apt install -y kubelet kubeadm kubectl
sudo apt-mark hold kubelet kubeadm kubectl
```

### 🔹 Red Hat 계열 (CentOS / Rocky Linux)
```sh
sudo yum install -y yum-utils device-mapper-persistent-data lvm2
sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
sudo yum install -y kubelet kubeadm kubectl --disableexcludes=kubernetes
sudo systemctl enable --now kubelet
```

### 🔹 macOS (Homebrew 이용)
```sh
brew install kubectl
```

---

## 3️⃣ Kubernetes 클러스터 구성

### 🔹 마스터 노드 초기화
```sh
sudo kubeadm init --pod-network-cidr=10.244.0.0/16
```

✅ 실행 후 아래 명령어를 따라 워커 노드를 추가합니다.
```sh
kubeadm join <마스터노드-IP>:6443 --token <TOKEN> --discovery-token-ca-cert-hash sha256:<HASH>
```

### 🔹 클러스터 사용을 위한 설정
```sh
mkdir -p $HOME/.kube
sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
sudo chown $(id -u):$(id -g) $HOME/.kube/config
```

---

## 4️⃣ 네트워크 플러그인 설치 (CNI)
Kubernetes 네트워크 설정을 위해 **Flannel** 또는 **Calico**를 사용할 수 있습니다.

### 🔹 Flannel 설치
```sh
kubectl apply -f https://github.com/flannel-io/flannel/releases/latest/download/kube-flannel.yml
```

### 🔹 Calico 설치
```sh
kubectl apply -f https://docs.projectcalico.org/manifests/calico.yaml
```

---

## 5️⃣ Kubernetes 애플리케이션 배포

### 🔹 Nginx 배포 예제
```sh
kubectl create deployment nginx --image=nginx --replicas=3
kubectl expose deployment nginx --type=NodePort --port=80
```

➡ `kubectl get svc` 명령어를 실행하여 NodePort를 확인한 후, `http://<노드-IP>:<포트>`에서 접근 가능합니다.

---

## 6️⃣ Helm 패키지 매니저 설치

Helm은 Kubernetes 애플리케이션을 배포하고 관리하는 패키지 매니저입니다.

```sh
curl https://raw.githubusercontent.com/helm/helm/master/scripts/get-helm-3 | bash
helm repo add stable https://charts.helm.sh/stable
helm repo update
```

✅ Helm을 활용한 Nginx 배포 예제
```sh
helm install my-nginx stable/nginx-ingress --set controller.replicaCount=2
```

---

## 7️⃣ Kubernetes 클러스터 관리

### 🔹 노드 상태 확인
```sh
kubectl get nodes
```

### 🔹 파드(Pod) 상태 확인
```sh
kubectl get pods -A
```

### 🔹 로그 확인
```sh
kubectl logs <POD_NAME>
```

### 🔹 서비스 삭제
```sh
kubectl delete deployment nginx
kubectl delete svc nginx
```

---

## 🎯 마무리
이 문서는 **Kubernetes 환경 설정 및 애플리케이션 배포**를 위한 가이드입니다. 추가적인 설정이나 확장이 필요하면 공식 문서를 참고하세요.

📖 [Kubernetes 공식 문서](https://kubernetes.io/docs/)

