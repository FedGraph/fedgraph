package common

import (
	"reflect"
	"testing"

	rayv1 "github.com/ray-project/kuberay/ray-operator/apis/ray/v1"

	"github.com/stretchr/testify/assert"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// Test subject and role ref names in the function BuildRoleBinding.
func TestBuildRoleBindingSubjectAndRoleRefName(t *testing.T) {
	tests := map[string]struct {
		input *rayv1.RayCluster
		want  []string
	}{
		"Ray cluster with head group service account": {
			input: &rayv1.RayCluster{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "raycluster-sample",
					Namespace: "default",
				},
				Spec: rayv1.RayClusterSpec{
					HeadGroupSpec: rayv1.HeadGroupSpec{
						Template: corev1.PodTemplateSpec{
							Spec: corev1.PodSpec{
								ServiceAccountName: "my-service-account",
							},
						},
					},
				},
			},
			want: []string{"my-service-account", "raycluster-sample"},
		},
		"Ray cluster without head group service account": {
			input: &rayv1.RayCluster{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "raycluster-sample",
					Namespace: "default",
				},
				Spec: rayv1.RayClusterSpec{
					HeadGroupSpec: rayv1.HeadGroupSpec{
						Template: corev1.PodTemplateSpec{
							Spec: corev1.PodSpec{},
						},
					},
				},
			},
			want: []string{"raycluster-sample", "raycluster-sample"},
		},
		"Ray cluster with a long name and without head group service account": {
			input: &rayv1.RayCluster{
				ObjectMeta: metav1.ObjectMeta{
					Name:      longString(t), // 200 chars long
					Namespace: "default",
				},
				Spec: rayv1.RayClusterSpec{
					HeadGroupSpec: rayv1.HeadGroupSpec{
						Template: corev1.PodTemplateSpec{
							Spec: corev1.PodSpec{},
						},
					},
				},
			},
			want: []string{
				shortString(t), // 50 chars long, truncated by utils.CheckName
				shortString(t),
			},
		},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			rb, err := BuildRoleBinding(tc.input)
			assert.Nil(t, err)
			got := []string{rb.Subjects[0].Name, rb.RoleRef.Name}
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("got %v, want %v", got, tc.want)
			}
		})
	}
}
