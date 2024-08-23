package common

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"reflect"
	"testing"

	rayv1 "github.com/ray-project/kuberay/ray-operator/apis/ray/v1"
	"github.com/ray-project/kuberay/ray-operator/controllers/ray/utils"

	"github.com/stretchr/testify/assert"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

var (
	headServiceAnnotationKey1   = "HeadServiceAnnotationKey1"
	headServiceAnnotationValue1 = "HeadServiceAnnotationValue1"
	headServiceAnnotationKey2   = "HeadServiceAnnotationKey2"
	headServiceAnnotationValue2 = "HeadServiceAnnotationValue2"
	serviceInstance             = &rayv1.RayService{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "rayservice-sample",
			Namespace: "default",
		},
		Spec: rayv1.RayServiceSpec{
			RayClusterSpec: rayv1.RayClusterSpec{
				HeadGroupSpec: rayv1.HeadGroupSpec{
					ServiceType: corev1.ServiceTypeClusterIP,
				},
			},
		},
	}
	instanceWithWrongSvc = &rayv1.RayCluster{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "raycluster-sample",
			Namespace: "default",
		},
		Spec: rayv1.RayClusterSpec{
			HeadServiceAnnotations: map[string]string{
				headServiceAnnotationKey1: headServiceAnnotationValue1,
				headServiceAnnotationKey2: headServiceAnnotationValue2,
			},
			HeadGroupSpec: rayv1.HeadGroupSpec{
				RayStartParams: map[string]string{
					"port":                "6379",
					"object-manager-port": "12345",
					"node-manager-port":   "12346",
					"object-store-memory": "100000000",
					"num-cpus":            "1",
				},
				Template: corev1.PodTemplateSpec{
					ObjectMeta: metav1.ObjectMeta{
						Namespace: "default",
						Labels: map[string]string{
							"groupName": utils.RayNodeHeadGroupLabelValue,
						},
					},
					Spec: corev1.PodSpec{
						Containers: []corev1.Container{
							{
								Name:  "ray-head",
								Image: "rayproject/autoscaler",
								Ports: []corev1.ContainerPort{
									{
										ContainerPort: 6379,
										Name:          "gcs",
									},
									{
										ContainerPort: 8265,
									},
									{
										ContainerPort: 8000,
										Name:          "serve",
									},
								},
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
	instanceForServeSvc = &rayv1.RayCluster{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "raycluster-sample-svc",
			Namespace: "default",
		},
		Spec: rayv1.RayClusterSpec{
			HeadServiceAnnotations: map[string]string{
				headServiceAnnotationKey1: headServiceAnnotationValue1,
				headServiceAnnotationKey2: headServiceAnnotationValue2,
			},
			HeadGroupSpec: rayv1.HeadGroupSpec{
				ServiceType: corev1.ServiceTypeClusterIP,
				Template: corev1.PodTemplateSpec{
					Spec: corev1.PodSpec{
						Containers: []corev1.Container{
							{
								Name: "ray-head",
								Ports: []corev1.ContainerPort{
									{ContainerPort: 8000, Name: "serve"},
								},
							},
						},
					},
				},
			},
		},
	}
)

func TestBuildServiceForHeadPod(t *testing.T) {
	svc, err := BuildServiceForHeadPod(context.Background(), *instanceWithWrongSvc, nil, nil)
	assert.Nil(t, err)

	actualResult := svc.Spec.Selector[utils.RayClusterLabelKey]
	expectedResult := instanceWithWrongSvc.Name
	if !reflect.DeepEqual(expectedResult, actualResult) {
		t.Fatalf("Expected `%v` but got `%v`", expectedResult, actualResult)
	}

	actualResult = svc.Spec.Selector[utils.RayNodeTypeLabelKey]
	expectedResult = string(rayv1.HeadNode)
	if !reflect.DeepEqual(expectedResult, actualResult) {
		t.Fatalf("Expected `%v` but got `%v`", expectedResult, actualResult)
	}

	actualResult = svc.Spec.Selector[utils.KubernetesApplicationNameLabelKey]
	expectedResult = utils.ApplicationName
	if !reflect.DeepEqual(expectedResult, actualResult) {
		t.Fatalf("Expected `%v` but got `%v`", expectedResult, actualResult)
	}

	ports := svc.Spec.Ports
	expectedResult = utils.DefaultServiceAppProtocol
	for _, port := range ports {
		if *port.AppProtocol != utils.DefaultServiceAppProtocol {
			t.Fatalf("Expected `%v` but got `%v`", expectedResult, *port.AppProtocol)
		}
	}
	// BuildServiceForHeadPod should generate a headless service for a Head Pod by default.
	if svc.Spec.ClusterIP != corev1.ClusterIPNone {
		t.Fatalf("Expected `%v` but got `%v`", corev1.ClusterIPNone, svc.Spec.ClusterIP)
	}
}

func TestBuildClusterIPServiceForHeadPod(t *testing.T) {
	os.Setenv(utils.ENABLE_RAY_HEAD_CLUSTER_IP_SERVICE, "true")
	defer os.Unsetenv(utils.ENABLE_RAY_HEAD_CLUSTER_IP_SERVICE)
	svc, err := BuildServiceForHeadPod(context.Background(), *instanceWithWrongSvc, nil, nil)
	assert.Nil(t, err)
	// BuildServiceForHeadPod should not generate a headless service for a Head Pod if ENABLE_RAY_HEAD_CLUSTER_IP_SERVICE is set.
	if svc.Spec.ClusterIP == corev1.ClusterIPNone {
		t.Fatalf("Not expected `%v` but got `%v`", corev1.ClusterIPNone, svc.Spec.ClusterIP)
	}
}

func TestBuildServiceForHeadPodWithAppNameLabel(t *testing.T) {
	labels := make(map[string]string)
	labels[utils.KubernetesApplicationNameLabelKey] = "testname"

	svc, err := BuildServiceForHeadPod(context.Background(), *instanceWithWrongSvc, labels, nil)
	assert.Nil(t, err)

	actualResult := svc.Spec.Selector[utils.KubernetesApplicationNameLabelKey]
	expectedResult := "testname"
	if !reflect.DeepEqual(expectedResult, actualResult) {
		t.Fatalf("Expected `%v` but got `%v`", expectedResult, actualResult)
	}

	actualLength := len(svc.Spec.Selector)
	// We have 5 default labels in `BuildServiceForHeadPod`, and `KubernetesApplicationNameLabelKey`
	// is one of the default labels. Hence, `expectedLength` should also be 5.
	expectedLength := 5
	if actualLength != expectedLength {
		t.Fatalf("Expected `%v` but got `%v`", expectedLength, actualLength)
	}
}

func TestBuildServiceForHeadPodWithAnnotations(t *testing.T) {
	annotations := make(map[string]string)
	annotations["key1"] = "testvalue1"
	annotations["key2"] = "testvalue2"
	svc, err := BuildServiceForHeadPod(context.Background(), *instanceWithWrongSvc, nil, annotations)
	assert.Nil(t, err)

	if !reflect.DeepEqual(svc.ObjectMeta.Annotations, annotations) {
		t.Fatalf("Expected `%v` but got `%v`", annotations, svc.ObjectMeta.Annotations)
	}
}

func TestGetPortsFromCluster(t *testing.T) {
	svcPorts, err := getPortsFromCluster(*instanceWithWrongSvc)
	assert.Nil(t, err)

	// getPortsFromCluster creates service ports based on the container ports.
	// It will assign a generated service port name if the container port name
	// is not defined. To compare created service ports with container ports,
	// all generated service port names need to be reverted to empty strings.
	svcNames := map[int32]string{}
	for name, port := range svcPorts {
		if name == (fmt.Sprint(port) + "-port") {
			name = ""
		}
		svcNames[port] = name
	}

	cPorts := instanceWithWrongSvc.Spec.HeadGroupSpec.Template.Spec.Containers[utils.RayContainerIndex].Ports

	for _, cPort := range cPorts {
		expectedResult := cPort.Name
		actualResult := svcNames[cPort.ContainerPort]
		if !reflect.DeepEqual(expectedResult, actualResult) {
			t.Fatalf("Expected `%v` but got `%v`", expectedResult, actualResult)
		}
	}
}

func TestGetServicePortsWithMetricsPort(t *testing.T) {
	cluster := instanceWithWrongSvc.DeepCopy()

	// Test case 1: No ports are specified by the user.
	cluster.Spec.HeadGroupSpec.Template.Spec.Containers[0].Ports = []corev1.ContainerPort{}
	ports := getServicePorts(*cluster)
	// Verify that getServicePorts sets the default metrics port when the user doesn't specify any ports.
	if ports[utils.MetricsPortName] != int32(utils.DefaultMetricsPort) {
		t.Fatalf("Expected `%v` but got `%v`", int32(utils.DefaultMetricsPort), ports[utils.MetricsPortName])
	}

	// Test case 2: Only a random port is specified by the user.
	cluster.Spec.HeadGroupSpec.Template.Spec.Containers[0].Ports = []corev1.ContainerPort{
		{
			Name:          "random",
			ContainerPort: 1234,
		},
	}
	ports = getServicePorts(*cluster)
	// Verify that getServicePorts sets the default metrics port when the user doesn't specify the metrics port but specifies other ports.
	if ports[utils.MetricsPortName] != int32(utils.DefaultMetricsPort) {
		t.Fatalf("Expected `%v` but got `%v`", int32(utils.DefaultMetricsPort), ports[utils.MetricsPortName])
	}

	// Test case 3: A custom metrics port is specified by the user.
	customMetricsPort := int32(utils.DefaultMetricsPort) + 1
	metricsPort := corev1.ContainerPort{
		Name:          utils.MetricsPortName,
		ContainerPort: customMetricsPort,
	}
	cluster.Spec.HeadGroupSpec.Template.Spec.Containers[0].Ports = append(cluster.Spec.HeadGroupSpec.Template.Spec.Containers[0].Ports, metricsPort)
	ports = getServicePorts(*cluster)
	// Verify that getServicePorts uses the user's custom metrics port when the user specifies the metrics port.
	if ports[utils.MetricsPortName] != customMetricsPort {
		t.Fatalf("Expected `%v` but got `%v`", customMetricsPort, ports[utils.MetricsPortName])
	}
}

func TestUserSpecifiedHeadService(t *testing.T) {
	// Use any RayCluster instance as a base for the test.
	testRayClusterWithHeadService := instanceWithWrongSvc.DeepCopy()

	// Set user-specified head service with user-specified labels, annotations, and ports.
	userName := "user-custom-name"
	userNamespace := "user-custom-namespace"
	userLabels := map[string]string{"userLabelKey": "userLabelValue", utils.RayClusterLabelKey: "userClusterName"} // Override default cluster name
	userAnnotations := map[string]string{"userAnnotationKey": "userAnnotationValue", headServiceAnnotationKey1: "user_override"}
	userPort := corev1.ServicePort{Name: "userPort", Port: 12345}
	userPortOverride := corev1.ServicePort{Name: utils.ClientPortName, Port: 98765} // Override default client port (10001)
	userPorts := []corev1.ServicePort{userPort, userPortOverride}
	userSelector := map[string]string{"userSelectorKey": "userSelectorValue", utils.RayClusterLabelKey: "userSelectorClusterName"}
	// Specify a "LoadBalancer" type, which differs from the default "ClusterIP" type.
	userType := corev1.ServiceTypeLoadBalancer
	// Specify an empty ClusterIP, which differs from the default "None" used by the BuildServeServiceForRayService.
	userClusterIP := ""
	testRayClusterWithHeadService.Spec.HeadGroupSpec.HeadService = &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:        userName,
			Namespace:   userNamespace,
			Labels:      userLabels,
			Annotations: userAnnotations,
		},
		Spec: corev1.ServiceSpec{
			Ports:     userPorts,
			Selector:  userSelector,
			Type:      userType,
			ClusterIP: userClusterIP,
		},
	}
	// These labels originate from HeadGroupSpec.Template.ObjectMeta.Labels
	userTemplateClusterName := "userTemplateClusterName"
	templateLabels := map[string]string{utils.RayClusterLabelKey: userTemplateClusterName}
	headService, err := BuildServiceForHeadPod(context.Background(), *testRayClusterWithHeadService, templateLabels, testRayClusterWithHeadService.Spec.HeadServiceAnnotations)
	if err != nil {
		t.Errorf("failed to build head service: %v", err)
	}

	// BuildServiceForHeadPod should respect the ClusterIP specified by users.
	if headService.Spec.ClusterIP != userClusterIP {
		t.Fatalf("Expected `%v` but got `%v`", userClusterIP, headService.Spec.ClusterIP)
	}

	// The selector field should only use the keys from the five default labels.  The values should be updated with the values from the template labels.
	// The user-provided HeadService labels should be ignored for the purposes of the selector field. The user-provided Selector field should be ignored.
	defaultLabels := HeadServiceLabels(*testRayClusterWithHeadService)
	// Make sure this test isn't spuriously passing. Check that RayClusterLabelKey is in the default labels.
	if _, ok := defaultLabels[utils.RayClusterLabelKey]; !ok {
		t.Errorf("utils.RayClusterLabelKey=%s should be in the default labels", utils.RayClusterLabelKey)
	}
	for k, v := range headService.Spec.Selector {
		// If k is not in the default labels, then the selector field should not contain it.
		if _, ok := defaultLabels[k]; !ok {
			t.Errorf("Selector field should not contain key=%s", k)
		}
		// If k is in the template labels, then the selector field should contain it with the value from the template labels.
		// Otherwise, it should contain the value from the default labels.
		if _, ok := templateLabels[k]; ok {
			if v != templateLabels[k] {
				t.Errorf("Selector field should contain key=%s with value=%s, actual value=%s", k, templateLabels[k], v)
			}
		} else {
			if v != defaultLabels[k] {
				t.Errorf("Selector field should contain key=%s with value=%s, actual value=%s", k, defaultLabels[k], v)
			}
		}
	}
	// The selector field should have every key from the default labels.
	for k := range defaultLabels {
		if _, ok := headService.Spec.Selector[k]; !ok {
			t.Errorf("Selector field should contain key=%s", k)
		}
	}

	// Print default labels for debugging
	for k, v := range defaultLabels {
		fmt.Printf("default label: key=%s, value=%s\n", k, v)
	}

	// Test merged labels. The final labels (headService.ObjectMeta.Labels) should consist of:
	// 1. The final selector (headService.Spec.Selector), updated with
	// 2. The user-specified labels from the HeadService (userLabels).
	// In the case of overlap, the selector labels have priority over userLabels.
	for k, v := range headService.ObjectMeta.Labels {
		// If k is in the user-specified labels, then the final labels should contain it with the value from the final selector.
		// Otherwise, it should contain the value from userLabels from the HeadService.
		if _, ok := headService.Spec.Selector[k]; ok {
			if v != headService.Spec.Selector[k] {
				t.Errorf("Final labels should contain key=%s with value=%s, actual value=%s", k, headService.Spec.Selector[k], v)
			}
		} else if _, ok := userLabels[k]; ok {
			if v != userLabels[k] {
				t.Errorf("Final labels should contain key=%s with value=%s, actual value=%s", k, userLabels[k], v)
			}
		} else {
			t.Errorf("Final labels contains key=%s but it should not", k)
		}
	}
	// Check that every key from the final selector (headService.Spec.Selector) and userLabels is in the final labels.
	for k := range headService.Spec.Selector {
		if _, ok := headService.ObjectMeta.Labels[k]; !ok {
			t.Errorf("Final labels should contain key=%s", k)
		}
	}

	// Test merged annotations. In the case of overlap (HeadServiceAnnotationKey1) the user annotation should be ignored.
	for k, v := range userAnnotations {
		if headService.ObjectMeta.Annotations[k] != v && k != headServiceAnnotationKey1 {
			t.Errorf("User annotation not found or incorrect value: key=%s, expected value=%s, actual value=%s", k, v, headService.ObjectMeta.Annotations[k])
		}
	}
	if headService.ObjectMeta.Annotations[headServiceAnnotationKey1] != headServiceAnnotationValue1 {
		t.Errorf("User annotation not found or incorrect value: key=%s, expected value=%s, actual value=%s", headServiceAnnotationKey1, headServiceAnnotationValue1, headService.ObjectMeta.Annotations[headServiceAnnotationKey1])
	}
	// HeadServiceAnnotationKey2 should be present with value HeadServiceAnnotationValue2 since it was only specified in HeadServiceAnnotations.
	if headService.ObjectMeta.Annotations[headServiceAnnotationKey2] != headServiceAnnotationValue2 {
		t.Errorf("User annotation not found or incorrect value: key=%s, expected value=%s, actual value=%s", headServiceAnnotationKey2, headServiceAnnotationValue2, headService.ObjectMeta.Annotations[headServiceAnnotationKey2])
	}

	// Test merged ports. In the case of overlap (ClientPortName) the user port should be ignored.
	// DEBUG: Print out the entire head service to help with debugging.
	headServiceJSON, err := json.MarshalIndent(headService, "", "  ")
	if err != nil {
		t.Errorf("failed to marshal head service: %v", err)
	}
	t.Logf("head service: %s", string(headServiceJSON))

	// Test merged ports
	for _, p := range userPorts {
		found := false
		for _, hp := range headService.Spec.Ports {
			if p.Name == hp.Name && p.Port == hp.Port {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("User port not found: %v", p)
		}
	}

	validateServiceTypeForUserSpecifiedService(headService, userType, t)
	validateLabelsForUserSpecifiedService(headService, userLabels, t)
	validateNameAndNamespaceForUserSpecifiedService(headService, testRayClusterWithHeadService.ObjectMeta.Namespace, userName, t)
}

func TestNilMapDoesntErrorInUserSpecifiedHeadService(t *testing.T) {
	// Use any RayCluster instance as a base for the test.
	testRayClusterWithHeadService := instanceWithWrongSvc.DeepCopy()

	// Set user-specified head service with many nil fields.
	testRayClusterWithHeadService.Spec.HeadGroupSpec.HeadService = &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{},
	}

	_, err := BuildServiceForHeadPod(context.Background(), *testRayClusterWithHeadService, nil, nil)
	if err != nil {
		t.Errorf("failed to build head service: %v", err)
	}
}

func TestBuildServiceForHeadPodPortsOrder(t *testing.T) {
	ctx := context.Background()
	svc1, err1 := BuildServiceForHeadPod(ctx, *instanceWithWrongSvc, nil, nil)
	svc2, err2 := BuildServiceForHeadPod(ctx, *instanceWithWrongSvc, nil, nil)
	assert.Nil(t, err1)
	assert.Nil(t, err2)

	ports1 := svc1.Spec.Ports
	ports2 := svc2.Spec.Ports

	// length should be same
	assert.Equal(t, len(ports1), len(ports2))
	for i := 0; i < len(ports1); i++ {
		// name should be same
		assert.Equal(t, ports1[i].Name, ports2[i].Name)
	}
}

func TestBuildServeServiceForRayService(t *testing.T) {
	svc, err := BuildServeServiceForRayService(context.Background(), *serviceInstance, *instanceWithWrongSvc)
	assert.Nil(t, err)

	actualResult := svc.Spec.Selector[utils.RayClusterLabelKey]
	expectedResult := instanceWithWrongSvc.Name
	if !reflect.DeepEqual(expectedResult, actualResult) {
		t.Fatalf("Expected `%v` but got `%v`", expectedResult, actualResult)
	}

	actualLabel := svc.Labels[utils.RayOriginatedFromCRNameLabelKey]
	expectedLabel := serviceInstance.Name
	if !reflect.DeepEqual(expectedLabel, actualLabel) {
		t.Fatalf("Expected `%v` but got `%v`", expectedLabel, actualLabel)
	}

	actualLabel = svc.Labels[utils.RayOriginatedFromCRDLabelKey]
	expectedLabel = utils.RayOriginatedFromCRDLabelValue(utils.RayServiceCRD)
	if !reflect.DeepEqual(expectedLabel, actualLabel) {
		t.Fatalf("Expected `%v` but got `%v`", expectedLabel, actualLabel)
	}

	actualType := svc.Spec.Type
	expectedType := corev1.ServiceTypeClusterIP
	if !reflect.DeepEqual(expectedType, actualType) {
		t.Fatalf("Expected `%v` but got `%v`", expectedType, actualType)
	}

	expectedName := fmt.Sprintf("%s-%s-%s", serviceInstance.Name, "serve", "svc")
	validateNameAndNamespaceForUserSpecifiedService(svc, serviceInstance.ObjectMeta.Namespace, expectedName, t)
}

func TestBuildServeServiceForRayCluster(t *testing.T) {
	svc, err := BuildServeServiceForRayCluster(context.Background(), *instanceForServeSvc)
	assert.Nil(t, err)

	actualResult := svc.Spec.Selector[utils.RayClusterLabelKey]
	expectedResult := instanceForServeSvc.Name
	if !reflect.DeepEqual(expectedResult, actualResult) {
		t.Fatalf("Expected `%v` but got `%v`", expectedResult, actualResult)
	}

	actualLabel := svc.Labels[utils.RayOriginatedFromCRNameLabelKey]
	expectedLabel := instanceForServeSvc.Name
	assert.Equal(t, expectedLabel, actualLabel)

	actualLabel = svc.Labels[utils.RayOriginatedFromCRDLabelKey]
	expectedLabel = utils.RayOriginatedFromCRDLabelValue(utils.RayClusterCRD)
	assert.Equal(t, expectedLabel, actualLabel)

	actualType := svc.Spec.Type
	expectedType := instanceForServeSvc.Spec.HeadGroupSpec.ServiceType
	if !reflect.DeepEqual(expectedType, actualType) {
		t.Fatalf("Expected `%v` but got `%v`", expectedType, actualType)
	}

	expectedName := fmt.Sprintf("%s-%s-%s", instanceForServeSvc.Name, "serve", "svc")
	validateNameAndNamespaceForUserSpecifiedService(svc, serviceInstance.ObjectMeta.Namespace, expectedName, t)
}

func TestBuildServeServiceForRayService_WithoutServePort(t *testing.T) {
	// Create a RayCluster without a port with the name "serve" in the Ray head container.
	cluster := rayv1.RayCluster{
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
								Name: "ray-head",
								Ports: []corev1.ContainerPort{
									{ContainerPort: 6379, Name: "gcs"},
								},
							},
						},
					},
				},
			},
		},
	}
	svc, err := BuildServeServiceForRayService(context.Background(), *serviceInstance, cluster)
	assert.NotNil(t, err)
	assert.Nil(t, svc)
}

func TestUserSpecifiedServeService(t *testing.T) {
	// Use any RayService instance as a base for the test.
	testRayServiceWithServeService := serviceInstance.DeepCopy()

	userName := "user-custom-name"
	userNamespace := "user-custom-namespace"
	userLabels := map[string]string{"userLabelKey": "userLabelValue", utils.RayClusterLabelKey: "userClusterName"} // Override default cluster name
	userAnnotations := map[string]string{"userAnnotationKey": "userAnnotationValue", "userAnnotationKey2": "userAnnotationValue2"}
	userPort := corev1.ServicePort{Name: "serve", Port: 12345}
	userPortOverride := corev1.ServicePort{Name: utils.ClientPortName, Port: 98765} // Override default client port (10001)
	userPorts := []corev1.ServicePort{userPort, userPortOverride}
	userSelector := map[string]string{"userSelectorKey": "userSelectorValue", utils.RayClusterLabelKey: "userSelectorClusterName"}
	// Specify a "LoadBalancer" type, which differs from the default "ClusterIP" type.
	userType := corev1.ServiceTypeLoadBalancer

	testRayServiceWithServeService.Spec.ServeService = &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:        userName,
			Namespace:   userNamespace,
			Labels:      userLabels,
			Annotations: userAnnotations,
		},
		Spec: corev1.ServiceSpec{
			Ports:    userPorts,
			Selector: userSelector,
			Type:     userType,
		},
	}

	svc, err := BuildServeServiceForRayService(context.Background(), *testRayServiceWithServeService, *instanceWithWrongSvc)
	if err != nil {
		t.Errorf("failed to build serve service: %v", err)
	}

	// Check every annotation is in the service annotation
	for k := range userAnnotations {
		if _, ok := svc.ObjectMeta.Annotations[k]; !ok {
			t.Errorf("Final labels should contain key=%s", k)
		}
	}

	// Check that selectors only have default selectors
	if len(svc.Spec.Selector) != 2 {
		t.Errorf("Selectors should have just 2 keys %s and %s", utils.RayClusterLabelKey, utils.RayClusterServingServiceLabelKey)
	}
	if svc.Spec.Selector[utils.RayClusterLabelKey] != instanceWithWrongSvc.Name {
		t.Errorf("Serve Service selector key %s value didn't match expected value : expected value=%s, actual value=%s", utils.RayClusterLabelKey, instanceWithWrongSvc.Name, svc.Spec.Selector[utils.RayClusterLabelKey])
	}
	if svc.Spec.Selector[utils.RayClusterServingServiceLabelKey] != utils.EnableRayClusterServingServiceTrue {
		t.Errorf("Serve Service selector key %s value didn't match expected value : expected value=%s, actual value=%s", utils.RayClusterServingServiceLabelKey, utils.EnableRayClusterServingServiceTrue, svc.Spec.Selector[utils.RayClusterServingServiceLabelKey])
	}

	// ports should only have DefaultServePort
	ports := svc.Spec.Ports
	expectedPortName := utils.ServingPortName
	expectedPortNumber := int32(8000)
	for _, port := range ports {
		if port.Name != utils.ServingPortName {
			t.Fatalf("Expected `%v` but got `%v`", expectedPortName, port.Name)
		}
		if port.Port != expectedPortNumber {
			t.Fatalf("Expected `%v` but got `%v`", expectedPortNumber, port.Port)
		}
	}

	validateServiceTypeForUserSpecifiedService(svc, userType, t)
	validateLabelsForUserSpecifiedService(svc, userLabels, t)
	validateNameAndNamespaceForUserSpecifiedService(svc, testRayServiceWithServeService.ObjectMeta.Namespace, userName, t)
}

func validateServiceTypeForUserSpecifiedService(svc *corev1.Service, userType corev1.ServiceType, t *testing.T) {
	// Test that the user service type takes priority over the default service type (example: ClusterIP)
	if svc.Spec.Type != userType {
		t.Errorf("Generated service type is not %s", userType)
	}
}

func validateNameAndNamespaceForUserSpecifiedService(svc *corev1.Service, defaultNamespace string, userName string, t *testing.T) {
	// Test name and namespace are generated if not specified
	if svc.ObjectMeta.Name == "" {
		t.Errorf("Generated service name is empty")
	}
	if svc.ObjectMeta.Namespace == "" {
		t.Errorf("Generated service namespace is empty")
	}
	// The user-provided namespace should be ignored, but the name should be respected
	if svc.ObjectMeta.Namespace != defaultNamespace {
		t.Errorf("User-provided namespace should be ignored: expected namespace=%s, actual namespace=%s", defaultNamespace, svc.ObjectMeta.Namespace)
	}
	if svc.ObjectMeta.Name != userName {
		t.Errorf("User-provided name should be respected: expected name=%s, actual name=%s", userName, svc.ObjectMeta.Name)
	}
}

func validateLabelsForUserSpecifiedService(svc *corev1.Service, userLabels map[string]string, t *testing.T) {
	for k := range userLabels {
		if _, ok := svc.ObjectMeta.Labels[k]; !ok {
			t.Errorf("Final labels should contain key=%s", k)
		}
	}
}
