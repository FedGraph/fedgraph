package util

import (
	"reflect"
	"sort"
	"testing"

	api "github.com/ray-project/kuberay/proto/go_client"
	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

var sizelimit = resource.MustParse("100Gi")

var testVolume = &api.Volume{
	Name:       "hdfs",
	VolumeType: api.Volume_HOST_PATH,
	Source:     "/opt/hdfs",
	MountPath:  "/mnt/hdfs",
	ReadOnly:   true,
}

// There is only an fake case for test both MountPropagationMode and file type
// in real case hostToContainer mode may only valid for directory
var testFileVolume = &api.Volume{
	Name:                 "test-file",
	VolumeType:           api.Volume_HOST_PATH,
	MountPropagationMode: api.Volume_HOSTTOCONTAINER,
	Source:               "/root/proc/stat",
	MountPath:            "/proc/stat",
	HostPathType:         api.Volume_FILE,
	ReadOnly:             true,
}

var testPVCVolume = &api.Volume{
	Name:       "test-pvc",
	VolumeType: api.Volume_PERSISTENT_VOLUME_CLAIM,
	Source:     "my-pvc",
	MountPath:  "/pvc/dir",
	ReadOnly:   true,
}

var testEphemeralVolume = &api.Volume{
	Name:       "test-ephemeral",
	VolumeType: api.Volume_EPHEMERAL,
	MountPath:  "/ephimeral/dir",
	Storage:    "10Gi",
}

var testConfigMapVolume = &api.Volume{
	Name:       "configMap",
	MountPath:  "/tmp/configmap",
	VolumeType: api.Volume_CONFIGMAP,
	Source:     "my-config-map",
	Items: map[string]string{
		"key1": "path1",
		"key2": "path2",
	},
}

var testSecretVolume = &api.Volume{
	Name:       "secret",
	MountPath:  "/tmp/secret",
	VolumeType: api.Volume_SECRET,
	Source:     "my-secret",
}

var testEmptyDirVolume = &api.Volume{
	Name:       "emptyDir",
	MountPath:  "/tmp/emptydir",
	VolumeType: api.Volume_EMPTY_DIR,
	Storage:    "100Gi",
}

var testAutoscalerOptions = api.AutoscalerOptions{
	IdleTimeoutSeconds: 25,
	UpscalingMode:      "Default",
	ImagePullPolicy:    "Always",
	Envs: &api.EnvironmentVariables{
		Values: map[string]string{
			"AWS_KEY": "123",
		},
		ValuesFrom: map[string]*api.EnvValueFrom{
			"REDIS_PASSWORD": {
				Source: api.EnvValueFrom_SECRET,
				Name:   "redis-password-secret",
				Key:    "password",
			},
			"CONFIGMAP": {
				Source: api.EnvValueFrom_CONFIGMAP,
				Name:   "special-config",
				Key:    "special.how",
			},
			"ResourceFieldRef": {
				Source: api.EnvValueFrom_RESOURCEFIELD,
				Name:   "my-container",
				Key:    "resource",
			},
		},
	},
	Volumes: []*api.Volume{
		{
			Name:       "configMap",
			MountPath:  "/tmp/configmap",
			VolumeType: api.Volume_CONFIGMAP,
			Source:     "my-config-map",
			Items: map[string]string{
				"key1": "path1",
				"key2": "path2",
			},
		},
		{
			Name:       "secret",
			MountPath:  "/tmp/secret",
			VolumeType: api.Volume_SECRET,
			Source:     "my-secret",
		},
	},
	Cpu: "300m",
}

// Spec for testing
var headGroup = api.HeadGroupSpec{
	ComputeTemplate: "foo",
	Image:           "bar",
	ImagePullPolicy: "Always",
	ServiceType:     "ClusterIP",
	RayStartParams: map[string]string{
		"dashboard-host":      "0.0.0.0",
		"metrics-export-port": "8080",
		"num-cpus":            "0",
	},
	ServiceAccount:  "account",
	ImagePullSecret: "foo",
	EnableIngress:   true,
	Environment: &api.EnvironmentVariables{
		Values: map[string]string{
			"AWS_KEY": "123",
		},
		ValuesFrom: map[string]*api.EnvValueFrom{
			"REDIS_PASSWORD": {
				Source: api.EnvValueFrom_SECRET,
				Name:   "redis-password-secret",
				Key:    "password",
			},
			"CONFIGMAP": {
				Source: api.EnvValueFrom_CONFIGMAP,
				Name:   "special-config",
				Key:    "special.how",
			},
			"ResourceFieldRef": {
				Source: api.EnvValueFrom_RESOURCEFIELD,
				Name:   "my-container",
				Key:    "resource",
			},
			"FieldRef": {
				Source: api.EnvValueFrom_FIELD,
				Key:    "path",
			},
		},
	},
	Annotations: map[string]string{
		"foo": "bar",
	},
	Labels: map[string]string{
		"foo": "bar",
	},
}

var workerGroup = api.WorkerGroupSpec{
	GroupName:       "wg",
	ComputeTemplate: "foo",
	Image:           "bar",
	ImagePullPolicy: "Always",
	Replicas:        5,
	MinReplicas:     5,
	MaxReplicas:     5,
	RayStartParams: map[string]string{
		"node-ip-address": "$MY_POD_IP",
	},
	ServiceAccount:  "account",
	ImagePullSecret: "foo",
	Environment: &api.EnvironmentVariables{
		Values: map[string]string{
			"foo": "bar",
		},
	},
	Annotations: map[string]string{
		"foo": "bar",
	},
	Labels: map[string]string{
		"foo": "bar",
	},
}

var rayCluster = api.Cluster{
	Name:      "test_cluster",
	Namespace: "foo",
	Annotations: map[string]string{
		"kubernetes.io/ingress.class": "nginx",
	},
	ClusterSpec: &api.ClusterSpec{
		HeadGroupSpec: &headGroup,
		WorkerGroupSpec: []*api.WorkerGroupSpec{
			&workerGroup,
		},
	},
}

var rayClusterAutoScaler = api.Cluster{
	Name:      "test_cluster",
	Namespace: "foo",
	Annotations: map[string]string{
		"kubernetes.io/ingress.class": "nginx",
	},
	ClusterSpec: &api.ClusterSpec{
		HeadGroupSpec: &headGroup,
		WorkerGroupSpec: []*api.WorkerGroupSpec{
			&workerGroup,
		},
		EnableInTreeAutoscaling: true,
		AutoscalerOptions: &api.AutoscalerOptions{
			UpscalingMode:      "Default",
			IdleTimeoutSeconds: 120,
			ImagePullPolicy:    "IfNotPresent",
		},
	},
}

var template = api.ComputeTemplate{
	Name:      "",
	Namespace: "",
	Cpu:       2,
	Memory:    8,
	Tolerations: []*api.PodToleration{
		{
			Key:      "blah1",
			Operator: "Exists",
			Effect:   "NoExecute",
		},
	},
}

var expectedToleration = corev1.Toleration{
	Key:      "blah1",
	Operator: "Exists",
	Effect:   "NoExecute",
}

var expectedLabels = map[string]string{
	"foo": "bar",
}

var expectedHeadNodeEnv = []corev1.EnvVar{
	{
		Name: "MY_POD_IP",
		ValueFrom: &corev1.EnvVarSource{
			FieldRef: &corev1.ObjectFieldSelector{
				FieldPath: "status.podIP",
			},
		},
	},
	{
		Name:  "AWS_KEY",
		Value: "123",
	},
	{
		Name: "REDIS_PASSWORD",
		ValueFrom: &corev1.EnvVarSource{
			SecretKeyRef: &corev1.SecretKeySelector{
				LocalObjectReference: corev1.LocalObjectReference{
					Name: "redis-password-secret",
				},
				Key: "password",
			},
		},
	},
	{
		Name: "CONFIGMAP",
		ValueFrom: &corev1.EnvVarSource{
			ConfigMapKeyRef: &corev1.ConfigMapKeySelector{
				LocalObjectReference: corev1.LocalObjectReference{
					Name: "special-config",
				},
				Key: "special.how",
			},
		},
	},
	{
		Name: "ResourceFieldRef",
		ValueFrom: &corev1.EnvVarSource{
			ResourceFieldRef: &corev1.ResourceFieldSelector{
				ContainerName: "my-container",
				Resource:      "resource",
			},
		},
	},
	{
		Name: "FieldRef",
		ValueFrom: &corev1.EnvVarSource{
			FieldRef: &corev1.ObjectFieldSelector{
				FieldPath: "path",
			},
		},
	},
}

func TestBuildVolumes(t *testing.T) {
	targetVolume := corev1.Volume{
		Name: testVolume.Name,
		VolumeSource: corev1.VolumeSource{
			HostPath: &corev1.HostPathVolumeSource{
				Path: testVolume.Source,
				Type: newHostPathType(string(corev1.HostPathDirectory)),
			},
		},
	}
	targetFileVolume := corev1.Volume{
		Name: testFileVolume.Name,
		VolumeSource: corev1.VolumeSource{
			HostPath: &corev1.HostPathVolumeSource{
				Path: testFileVolume.Source,
				Type: newHostPathType(string(corev1.HostPathFile)),
			},
		},
	}

	targetPVCVolume := corev1.Volume{
		Name: testPVCVolume.Name,
		VolumeSource: corev1.VolumeSource{
			PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
				ClaimName: testPVCVolume.Source,
				ReadOnly:  testPVCVolume.ReadOnly,
			},
		},
	}

	targetEphemeralVolume := corev1.Volume{
		Name: testEphemeralVolume.Name,
		VolumeSource: corev1.VolumeSource{
			Ephemeral: &corev1.EphemeralVolumeSource{
				VolumeClaimTemplate: &corev1.PersistentVolumeClaimTemplate{
					ObjectMeta: metav1.ObjectMeta{
						Labels: map[string]string{
							"app.kubernetes.io/managed-by": "kuberay-apiserver",
						},
					},
					Spec: corev1.PersistentVolumeClaimSpec{
						AccessModes: []corev1.PersistentVolumeAccessMode{
							corev1.ReadWriteOnce,
						},
						Resources: corev1.ResourceRequirements{
							Requests: corev1.ResourceList{
								corev1.ResourceStorage: resource.MustParse(testEphemeralVolume.Storage),
							},
						},
					},
				},
			},
		},
	}

	targetConfigMapVolume := corev1.Volume{
		Name: "configMap",
		VolumeSource: corev1.VolumeSource{
			ConfigMap: &corev1.ConfigMapVolumeSource{
				LocalObjectReference: corev1.LocalObjectReference{
					Name: "my-config-map",
				},
				Items: []corev1.KeyToPath{
					{
						Key:  "key1",
						Path: "path1",
					},
					{
						Key:  "key2",
						Path: "path2",
					},
				},
			},
		},
	}

	targetSecretVolume := corev1.Volume{
		Name: "secret",
		VolumeSource: corev1.VolumeSource{
			Secret: &corev1.SecretVolumeSource{
				SecretName: "my-secret",
			},
		},
	}

	targetEmptyDirVolume := corev1.Volume{
		Name: "emptyDir",
		VolumeSource: corev1.VolumeSource{
			EmptyDir: &corev1.EmptyDirVolumeSource{
				SizeLimit: &sizelimit,
			},
		},
	}

	tests := []struct {
		name      string
		apiVolume []*api.Volume
		expect    []corev1.Volume
	}{
		{
			"normal test",
			[]*api.Volume{
				testVolume, testFileVolume,
			},
			[]corev1.Volume{targetVolume, targetFileVolume},
		},
		{
			"pvc test",
			[]*api.Volume{testPVCVolume},
			[]corev1.Volume{targetPVCVolume},
		},
		{
			"ephemeral test",
			[]*api.Volume{testEphemeralVolume},
			[]corev1.Volume{targetEphemeralVolume},
		},
		{
			"configmap test",
			[]*api.Volume{testConfigMapVolume},
			[]corev1.Volume{targetConfigMapVolume},
		},
		{
			"secret test",
			[]*api.Volume{testSecretVolume},
			[]corev1.Volume{targetSecretVolume},
		},
		{
			"empty dir test",
			[]*api.Volume{testEmptyDirVolume},
			[]corev1.Volume{targetEmptyDirVolume},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := buildVols(tt.apiVolume)
			assert.Nil(t, err)
			if tt.name == "configmap test" {
				// Sort items for comparison
				sort.SliceStable(got[0].ConfigMap.Items, func(i, j int) bool {
					return got[0].ConfigMap.Items[i].Key < got[0].ConfigMap.Items[j].Key
				})
				sort.SliceStable(tt.expect[0].ConfigMap.Items, func(i, j int) bool {
					return tt.expect[0].ConfigMap.Items[i].Key < tt.expect[0].ConfigMap.Items[j].Key
				})
			}
			if !reflect.DeepEqual(got, tt.expect) {
				t.Errorf("failed for %s ..., got %v, expected %v", tt.name, got, tt.expect)
			}
		})
	}
}

func TestBuildVolumeMounts(t *testing.T) {
	hostToContainer := corev1.MountPropagationHostToContainer
	targetVolumeMount := corev1.VolumeMount{
		Name:      testVolume.Name,
		ReadOnly:  testVolume.ReadOnly,
		MountPath: testVolume.MountPath,
	}
	targetFileVolumeMount := corev1.VolumeMount{
		Name:             testFileVolume.Name,
		ReadOnly:         testFileVolume.ReadOnly,
		MountPath:        testFileVolume.MountPath,
		MountPropagation: &hostToContainer,
	}
	targetPVCVolumeMount := corev1.VolumeMount{
		Name:      testPVCVolume.Name,
		ReadOnly:  testPVCVolume.ReadOnly,
		MountPath: testPVCVolume.MountPath,
	}
	tests := []struct {
		name      string
		apiVolume []*api.Volume
		expect    []corev1.VolumeMount
	}{
		{
			"normal test",
			[]*api.Volume{
				testVolume,
				testFileVolume,
			},
			[]corev1.VolumeMount{
				targetVolumeMount,
				targetFileVolumeMount,
			},
		},
		{
			"pvc test",
			[]*api.Volume{testPVCVolume},
			[]corev1.VolumeMount{targetPVCVolumeMount},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := buildVolumeMounts(tt.apiVolume)
			if !reflect.DeepEqual(got, tt.expect) {
				t.Errorf("failed for %s ..., got %v, expected %v", tt.name, got, tt.expect)
			}
		})
	}
}

func TestBuildHeadPodTemplate(t *testing.T) {
	podSpec, err := buildHeadPodTemplate("2.4", &api.EnvironmentVariables{}, &headGroup, &template, false)
	assert.Nil(t, err)

	if podSpec.Spec.ServiceAccountName != "account" {
		t.Errorf("failed to propagate service account")
	}
	if podSpec.Spec.ImagePullSecrets[0].Name != "foo" {
		t.Errorf("failed to propagate image pull secret")
	}
	if (string)(podSpec.Spec.Containers[0].ImagePullPolicy) != "Always" {
		t.Errorf("failed to propagate image pull policy")
	}
	if len(podSpec.Spec.Containers[0].Env) != 6 {
		t.Errorf("failed to propagate environment")
	}
	if len(podSpec.Spec.Containers[0].Ports) != 4 {
		t.Errorf("failed build ports")
	}
	// Sort values for comparison
	sort.SliceStable(podSpec.Spec.Containers[0].Env, func(i, j int) bool {
		return podSpec.Spec.Containers[0].Env[i].Name < podSpec.Spec.Containers[0].Env[j].Name
	})
	sort.SliceStable(expectedHeadNodeEnv, func(i, j int) bool {
		return expectedHeadNodeEnv[i].Name < expectedHeadNodeEnv[j].Name
	})

	if !reflect.DeepEqual(podSpec.Spec.Containers[0].Env, expectedHeadNodeEnv) {
		t.Errorf("failed to convert environment, got %v, expected %v", podSpec.Spec.Containers[0].Env, expectedHeadNodeEnv)
	}

	if len(podSpec.Spec.Tolerations) != 1 {
		t.Errorf("failed to propagate tolerations, expected 1, got %d", len(podSpec.Spec.Tolerations))
	}
	if !reflect.DeepEqual(podSpec.Spec.Tolerations[0], expectedToleration) {
		t.Errorf("failed to propagate annotations, got %v, expected %v", tolerationToString(&podSpec.Spec.Tolerations[0]),
			tolerationToString(&expectedToleration))
	}
	if val, exists := podSpec.Annotations["foo"]; !exists || val != "bar" {
		t.Errorf("failed to convert annotations")
	}
	if !reflect.DeepEqual(podSpec.Labels, expectedLabels) {
		t.Errorf("failed to convert labels, got %v, expected %v", podSpec.Labels, expectedLabels)
	}

	podSpec, err = buildHeadPodTemplate("2.4", &api.EnvironmentVariables{}, &headGroup, &template, true)
	assert.Nil(t, err)
	if len(podSpec.Spec.Containers[0].Ports) != 6 {
		t.Errorf("failed build ports")
	}
}

func TestConvertAutoscalerOptions(t *testing.T) {
	options, err := buildAutoscalerOptions(&testAutoscalerOptions)
	assert.Nil(t, err)
	assert.Equal(t, *options.IdleTimeoutSeconds, int32(25))
	assert.Equal(t, (string)(*options.UpscalingMode), "Default")
	assert.Equal(t, (string)(*options.ImagePullPolicy), "Always")
	assert.Equal(t, len(options.Env), 1)
	assert.Equal(t, len(options.EnvFrom), 2)
	assert.Equal(t, len(options.VolumeMounts), 2)
	assert.Equal(t, options.Resources.Requests.Cpu().String(), "300m")
	assert.Equal(t, options.Resources.Requests.Memory().String(), "512Mi")
}

func TestBuildRayCluster(t *testing.T) {
	cluster, err := NewRayCluster(&rayCluster, map[string]*api.ComputeTemplate{"foo": &template})
	assert.Nil(t, err)
	if len(cluster.ObjectMeta.Annotations) != 1 {
		t.Errorf("failed to propagate annotations")
	}
	if !(*cluster.Spec.HeadGroupSpec.EnableIngress) {
		t.Errorf("failed to propagate create Ingress")
	}
	assert.Equal(t, cluster.Spec.EnableInTreeAutoscaling, (*bool)(nil))
	cluster, err = NewRayCluster(&rayClusterAutoScaler, map[string]*api.ComputeTemplate{"foo": &template})
	assert.Nil(t, err)
	assert.Equal(t, *cluster.Spec.EnableInTreeAutoscaling, true)
	assert.NotEqual(t, cluster.Spec.AutoscalerOptions, nil)
}

func TestBuilWorkerPodTemplate(t *testing.T) {
	podSpec, err := buildWorkerPodTemplate("2.4", &api.EnvironmentVariables{}, &workerGroup, &template)
	assert.Nil(t, err)

	if podSpec.Spec.ServiceAccountName != "account" {
		t.Errorf("failed to propagate service account")
	}
	if podSpec.Spec.ImagePullSecrets[0].Name != "foo" {
		t.Errorf("failed to propagate image pull secret")
	}
	if (string)(podSpec.Spec.Containers[0].ImagePullPolicy) != "Always" {
		t.Errorf("failed to propagate image pull policy")
	}
	if !containsEnv(podSpec.Spec.Containers[0].Env, "foo", "bar") {
		t.Errorf("failed to propagate environment")
	}
	if len(podSpec.Spec.Tolerations) != 1 {
		t.Errorf("failed to propagate tolerations, expected 1, got %d", len(podSpec.Spec.Tolerations))
	}
	if !reflect.DeepEqual(podSpec.Spec.Tolerations[0], expectedToleration) {
		t.Errorf("failed to propagate annotations, got %v, expected %v", tolerationToString(&podSpec.Spec.Tolerations[0]),
			tolerationToString(&expectedToleration))
	}
	if val, exists := podSpec.Annotations["foo"]; !exists || val != "bar" {
		t.Errorf("failed to convert annotations")
	}
	if !reflect.DeepEqual(podSpec.Labels, expectedLabels) {
		t.Errorf("failed to convert labels, got %v, expected %v", podSpec.Labels, expectedLabels)
	}
}

func containsEnv(envs []corev1.EnvVar, key string, val string) bool {
	for _, env := range envs {
		if env.Name == key && env.Value == val {
			return true
		}
	}
	return false
}

func tolerationToString(toleration *corev1.Toleration) string {
	return "Key: " + toleration.Key + " Operator: " + string(toleration.Operator) + " Effect: " + string(toleration.Effect)
}
