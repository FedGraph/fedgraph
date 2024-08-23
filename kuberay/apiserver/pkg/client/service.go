package client

import (
	"time"

	klog "k8s.io/klog/v2"

	"github.com/ray-project/kuberay/apiserver/pkg/util"
	"sigs.k8s.io/controller-runtime/pkg/client/config"

	rayclient "github.com/ray-project/kuberay/ray-operator/pkg/client/clientset/versioned"
	rayv1 "github.com/ray-project/kuberay/ray-operator/pkg/client/clientset/versioned/typed/ray/v1"
)

type ServiceClientInterface interface {
	RayServiceClient(namespace string) rayv1.RayServiceInterface
}

type RayServiceClient struct {
	client rayv1.RayV1Interface
}

func (cc RayServiceClient) RayServiceClient(namespace string) rayv1.RayServiceInterface {
	return cc.client.RayServices(namespace)
}

func NewRayServiceClientOrFatal(initConnectionTimeout time.Duration, options util.ClientOptions) ServiceClientInterface {
	cfg, err := config.GetConfig()
	if err != nil {
		klog.Fatalf("Failed to create RayService client. Error: %v", err)
	}
	cfg.QPS = options.QPS
	cfg.Burst = options.Burst

	rayServiceClient := rayclient.NewForConfigOrDie(cfg).RayV1()
	return &RayServiceClient{client: rayServiceClient}
}
