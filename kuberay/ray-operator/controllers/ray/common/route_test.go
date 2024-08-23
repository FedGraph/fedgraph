package common

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/utils/pointer"

	rayv1 "github.com/ray-project/kuberay/ray-operator/apis/ray/v1"
)

var instanceWithRouteEnabled = &rayv1.RayCluster{
	ObjectMeta: metav1.ObjectMeta{
		Name:      "raycluster-sample",
		Namespace: "default",
		Annotations: map[string]string{
			IngressClassAnnotationKey: "nginx",
		},
	},
	Spec: rayv1.RayClusterSpec{
		HeadGroupSpec: rayv1.HeadGroupSpec{
			EnableIngress: pointer.Bool(true),
			Template: corev1.PodTemplateSpec{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name:    "ray-head",
							Image:   "rayproject/autoscaler",
							Command: []string{"python"},
							Args:    []string{"/opt/code.py"},
						},
					},
				},
			},
		},
	},
}

func TestBuildRouteForHeadService(t *testing.T) {
	route, err := BuildRouteForHeadService(*instanceWithRouteEnabled)
	assert.Nil(t, err)

	// Test name
	var builder strings.Builder
	builder.WriteString(instanceWithIngressEnabled.ObjectMeta.Name)
	builder.WriteString("-head-route")
	if builder.String() != route.Name {
		t.Fatalf("Error generating Route name. Expected `%v` but got `%v`", builder.String(), route.Name)
	}
	// Test To subject
	expectedKind := "Service"
	if expectedKind != route.Spec.To.Kind {
		t.Fatalf("Error generating Route kind. Expected `%v` but got `%v`", expectedKind, route.Spec.To.Kind)
	}
	// Test Service name
	builder.Reset()
	builder.WriteString(instanceWithIngressEnabled.ObjectMeta.Name)
	builder.WriteString("-head-svc")
	if builder.String() != route.Spec.To.Name {
		t.Fatalf("Error generating service name. Expected `%v` but got `%v`", builder.String(), route.Spec.To.Name)
	}

	// Test Service port
	expectedPort := intstr.FromInt(8265)
	if route.Spec.Port.TargetPort != expectedPort {
		t.Fatalf("Error generating service port. Expected `%v` but got `%v`", expectedPort, route.Spec.Port.TargetPort)
	}
}
