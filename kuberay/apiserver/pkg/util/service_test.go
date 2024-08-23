package util

import (
	"testing"

	api "github.com/ray-project/kuberay/proto/go_client"

	"github.com/stretchr/testify/assert"
)

var apiServiceNoServe = &api.RayService{
	Name:        "test",
	Namespace:   "test",
	User:        "test",
	ClusterSpec: rayCluster.ClusterSpec,
}

var apiServiceV2 = &api.RayService{
	Name:                            "test",
	Namespace:                       "test",
	User:                            "test",
	ServeConfig_V2:                  "Fake Yaml file",
	ClusterSpec:                     rayCluster.ClusterSpec,
	ServiceUnhealthySecondThreshold: 100,
}

func TestBuildService(t *testing.T) {
	_, err := NewRayService(apiServiceNoServe, map[string]*api.ComputeTemplate{"foo": &template})
	assert.NotNil(t, err)
	if err.Error() != "serve configuration is not defined" {
		t.Errorf("wrong error returned")
	}
	got, err := NewRayService(apiServiceV2, map[string]*api.ComputeTemplate{"foo": &template})
	assert.Nil(t, err)
	if got.RayService.Spec.ServeConfigV2 == "" {
		t.Errorf("Got empty V2")
	}
	assert.NotNil(t, got.Spec.ServiceUnhealthySecondThreshold)
	assert.Nil(t, got.Spec.DeploymentUnhealthySecondThreshold)
}
