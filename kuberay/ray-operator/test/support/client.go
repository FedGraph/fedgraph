package support

import (
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"

	// Import all Kubernetes client auth plugins (e.g. Azure, GCP, OIDC, etc.)
	// to ensure that exec-entrypoint and run can make use of them.
	_ "k8s.io/client-go/plugin/pkg/client/auth"

	rayclient "github.com/ray-project/kuberay/ray-operator/pkg/client/clientset/versioned"
)

type Client interface {
	Core() kubernetes.Interface
	Ray() rayclient.Interface
	Dynamic() dynamic.Interface
}

type testClient struct {
	core    kubernetes.Interface
	ray     rayclient.Interface
	dynamic dynamic.Interface
}

var _ Client = (*testClient)(nil)

func (t *testClient) Core() kubernetes.Interface {
	return t.core
}

func (t *testClient) Ray() rayclient.Interface {
	return t.ray
}

func (t *testClient) Dynamic() dynamic.Interface {
	return t.dynamic
}

func newTestClient() (Client, error) {
	cfg, err := clientcmd.NewNonInteractiveDeferredLoadingClientConfig(
		clientcmd.NewDefaultClientConfigLoadingRules(),
		&clientcmd.ConfigOverrides{},
	).ClientConfig()
	if err != nil {
		return nil, err
	}

	kubeClient, err := kubernetes.NewForConfig(cfg)
	if err != nil {
		return nil, err
	}

	rayClient, err := rayclient.NewForConfig(cfg)
	if err != nil {
		return nil, err
	}

	dynamicClient, err := dynamic.NewForConfig(cfg)
	if err != nil {
		return nil, err
	}

	return &testClient{
		core:    kubeClient,
		ray:     rayClient,
		dynamic: dynamicClient,
	}, nil
}
