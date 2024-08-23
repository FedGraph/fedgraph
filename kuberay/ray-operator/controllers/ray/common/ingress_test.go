package common

import (
	"context"
	"reflect"
	"testing"

	"github.com/ray-project/kuberay/ray-operator/controllers/ray/utils"

	rayv1 "github.com/ray-project/kuberay/ray-operator/apis/ray/v1"

	"github.com/stretchr/testify/assert"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

var instanceWithIngressEnabled = &rayv1.RayCluster{
	ObjectMeta: metav1.ObjectMeta{
		Name:      "raycluster-sample",
		Namespace: "default",
		Annotations: map[string]string{
			IngressClassAnnotationKey: "nginx",
		},
	},
	Spec: rayv1.RayClusterSpec{
		HeadGroupSpec: rayv1.HeadGroupSpec{
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

var instanceWithIngressEnabledWithoutIngressClass = &rayv1.RayCluster{
	ObjectMeta: metav1.ObjectMeta{
		Name:      "raycluster-sample",
		Namespace: "default",
	},
	Spec: rayv1.RayClusterSpec{
		HeadGroupSpec: rayv1.HeadGroupSpec{
			Template: corev1.PodTemplateSpec{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name:    "ray-head",
							Image:   "rayproject/autoscaler",
							Command: []string{"python"},
							Args:    []string{"/opt/code.py"},
							Env: []corev1.EnvVar{
								{
									Name: "MY_POD_IP",
									ValueFrom: &corev1.EnvVarSource{
										FieldRef: &corev1.ObjectFieldSelector{
											FieldPath: "status.podIP",
										},
									},
								},
							},
						},
					},
				},
			},
		},
	},
}

// only throw warning message and rely on Kubernetes to assign default ingress class
func TestBuildIngressForHeadServiceWithoutIngressClass(t *testing.T) {
	ingress, err := BuildIngressForHeadService(context.Background(), *instanceWithIngressEnabledWithoutIngressClass)
	assert.NotNil(t, ingress)
	assert.Nil(t, err)
}

func TestBuildIngressForHeadService(t *testing.T) {
	ingress, err := BuildIngressForHeadService(context.Background(), *instanceWithIngressEnabled)
	assert.Nil(t, err)

	// check ingress.class annotation
	actualResult := ingress.Labels[utils.RayClusterLabelKey]
	expectedResult := instanceWithIngressEnabled.Name
	if !reflect.DeepEqual(expectedResult, actualResult) {
		t.Fatalf("Expected `%v` but got `%v`", expectedResult, actualResult)
	}

	// `annotations.kubernetes.io/ingress.class` was deprecated in Kubernetes 1.18,
	// and `spec.ingressClassName` is a replacement for this annotation. See
	// kubernetes.io/docs/concepts/services-networking/ingress/#deprecated-annotation
	// for more details.
	actualResult = ingress.Annotations[IngressClassAnnotationKey]
	expectedResult = ""
	if !reflect.DeepEqual(expectedResult, actualResult) {
		t.Fatalf("Expected `%v` but got `%v`", expectedResult, actualResult)
	}

	actualResult = *ingress.Spec.IngressClassName
	expectedResult = instanceWithIngressEnabled.Annotations[IngressClassAnnotationKey]
	if !reflect.DeepEqual(expectedResult, actualResult) {
		t.Fatalf("Expected `%v` but got `%v`", expectedResult, actualResult)
	}

	// rules count
	assert.Equal(t, 1, len(ingress.Spec.Rules))

	// paths count
	expectedPaths := 1 // dashboard only
	actualPaths := len(ingress.Spec.Rules[0].IngressRuleValue.HTTP.Paths)
	if !reflect.DeepEqual(expectedPaths, actualPaths) {
		t.Fatalf("Expected `%v` but got `%v`", expectedPaths, actualPaths)
	}

	// path names
	paths := ingress.Spec.Rules[0].IngressRuleValue.HTTP.Paths
	headSvcName, err := utils.GenerateHeadServiceName(utils.RayClusterCRD, instanceWithIngressEnabled.Spec, instanceWithIngressEnabled.Name)
	assert.Nil(t, err)
	for _, path := range paths {
		actualResult = path.Backend.Service.Name
		expectedResult = headSvcName

		if !reflect.DeepEqual(expectedResult, actualResult) {
			t.Fatalf("Expected `%v` but got `%v`", expectedResult, actualResult)
		}
	}
}
