package e2e

import (
	"net/http"
	"reflect"
	"testing"

	kuberayHTTP "github.com/ray-project/kuberay/apiserver/pkg/http"
	api "github.com/ray-project/kuberay/proto/go_client"
	"github.com/stretchr/testify/require"
)

// TestCreateTemplate sequentially iterates over the create compute endpoint
// to validate various input scenarios
func TestCreateTemplate(t *testing.T) {
	tCtx, err := NewEnd2EndTestingContext(t)
	require.NoError(t, err, "No error expected when creating testing context")

	t.Cleanup(func() {
		tCtx.DeleteComputeTemplate(t)
	})

	tests := []GenericEnd2EndTest[*api.CreateComputeTemplateRequest]{
		{
			Name: "Create a valid compute template",
			Input: &api.CreateComputeTemplateRequest{
				ComputeTemplate: &api.ComputeTemplate{
					Name:      tCtx.GetComputeTemplateName(),
					Namespace: tCtx.GetNamespaceName(),
					Cpu:       2,
					Memory:    4,
				},
				Namespace: tCtx.GetNamespaceName(),
			},
			ExpectedError: nil,
		},
		{
			Name: "Create an invalid template with no Name",
			Input: &api.CreateComputeTemplateRequest{
				ComputeTemplate: &api.ComputeTemplate{
					Name:      "",
					Namespace: tCtx.GetNamespaceName(),
					Cpu:       2,
					Memory:    4,
					Gpu:       0,
				},
				Namespace: tCtx.GetNamespaceName(),
			},
			ExpectedError: &kuberayHTTP.KuberayAPIServerClientError{
				HTTPStatusCode: http.StatusBadRequest,
			},
		},
		{
			Name: "Create an invalid template with different namespace",
			Input: &api.CreateComputeTemplateRequest{
				ComputeTemplate: &api.ComputeTemplate{
					Name:      tCtx.GetComputeTemplateName(),
					Namespace: "another",
					Cpu:       2,
					Memory:    4,
					Gpu:       0,
				},
				Namespace: tCtx.GetNamespaceName(),
			},
			ExpectedError: &kuberayHTTP.KuberayAPIServerClientError{
				HTTPStatusCode: http.StatusBadRequest,
			},
		},
		{
			Name: "Create an invalid template with zero cpu",
			Input: &api.CreateComputeTemplateRequest{
				ComputeTemplate: &api.ComputeTemplate{
					Name:      tCtx.GetComputeTemplateName(),
					Namespace: tCtx.GetNamespaceName(),
					Cpu:       0,
					Memory:    4,
				},
				Namespace: tCtx.GetNamespaceName(),
			},
			ExpectedError: &kuberayHTTP.KuberayAPIServerClientError{
				HTTPStatusCode: http.StatusBadRequest,
			},
		},
		{
			Name: "Create an invalid template with zero memory",
			Input: &api.CreateComputeTemplateRequest{
				ComputeTemplate: &api.ComputeTemplate{
					Name:      tCtx.GetComputeTemplateName(),
					Namespace: tCtx.GetNamespaceName(),
					Cpu:       2,
					Memory:    0,
				},
				Namespace: tCtx.GetNamespaceName(),
			},
			ExpectedError: &kuberayHTTP.KuberayAPIServerClientError{
				HTTPStatusCode: http.StatusBadRequest,
			},
		},
		{
			Name: "Create a duplicate invalid",
			Input: &api.CreateComputeTemplateRequest{
				ComputeTemplate: &api.ComputeTemplate{
					Name:      tCtx.GetComputeTemplateName(),
					Namespace: tCtx.GetNamespaceName(),
					Cpu:       2,
					Memory:    0,
				},
				Namespace: tCtx.GetNamespaceName(),
			},
			ExpectedError: &kuberayHTTP.KuberayAPIServerClientError{
				HTTPStatusCode: http.StatusBadRequest,
			},
		},
	}
	// Execute tests sequentially
	for _, tc := range tests {
		tc := tc // capture range variable
		t.Run(tc.Name, func(t *testing.T) {
			actualTemplate, actualRpcStatus, err := tCtx.GetRayApiServerClient().CreateComputeTemplate(tc.Input)
			if tc.ExpectedError != nil {
				require.EqualError(t, err, tc.ExpectedError.Error(), "Matching error expected")
				require.NotNil(t, actualRpcStatus, "A not nill RPC status is required")
			} else {
				require.NoError(t, err, "No error expected")
				require.Nil(t, actualRpcStatus, "No RPC status expected")
				require.Truef(t, reflect.DeepEqual(tc.Input.ComputeTemplate, actualTemplate), "Equal templates expected")
			}
		})
	}
}

// TestDeleteTemplate sequentially iterates over the delete compute template endpoint
// to validate various input scenarios
func TestDeleteTemplate(t *testing.T) {
	tCtx, err := NewEnd2EndTestingContext(t)
	require.NoError(t, err, "No error expected when creating testing context")

	tCtx.CreateComputeTemplate(t)

	tests := []GenericEnd2EndTest[*api.DeleteComputeTemplateRequest]{
		{
			Name: "Delete existing template",
			Input: &api.DeleteComputeTemplateRequest{
				Name:      tCtx.GetComputeTemplateName(),
				Namespace: tCtx.GetNamespaceName(),
			},
			ExpectedError: nil,
		},
		{
			Name: "Delete non existing template",
			Input: &api.DeleteComputeTemplateRequest{
				Name:      "another-template",
				Namespace: tCtx.GetNamespaceName(),
			},
			ExpectedError: &kuberayHTTP.KuberayAPIServerClientError{
				HTTPStatusCode: http.StatusNotFound,
			},
		},
		{
			Name: "Delete a template with no name",
			Input: &api.DeleteComputeTemplateRequest{
				Namespace: tCtx.GetNamespaceName(),
			},
			ExpectedError: &kuberayHTTP.KuberayAPIServerClientError{
				HTTPStatusCode: http.StatusBadRequest,
			},
		},
		{
			Name: "Delete a template with no namespace",
			Input: &api.DeleteComputeTemplateRequest{
				Name: "some-name",
			},
			ExpectedError: &kuberayHTTP.KuberayAPIServerClientError{
				HTTPStatusCode: http.StatusNotFound,
			},
		},
	}
	// Execute tests sequentially
	for _, tc := range tests {
		tc := tc // capture range variable
		t.Run(tc.Name, func(t *testing.T) {
			actualRpcStatus, err := tCtx.GetRayApiServerClient().DeleteComputeTemplate(tc.Input)
			if tc.ExpectedError != nil {
				require.EqualError(t, err, tc.ExpectedError.Error(), "Matching error expected")
				require.NotNil(t, actualRpcStatus, "A not nill RPC status is required")
			} else {
				require.NoError(t, err, "No error expected")
				require.Nil(t, actualRpcStatus, "No RPC status expected")
			}
		})
	}
}

// TestGetAllTheTemplates tests gets all compute templates endpoint
// to validate various input scenarios
func TestGetAllComputeTemplates(t *testing.T) {
	tCtx, err := NewEnd2EndTestingContext(t)
	require.NoError(t, err, "No error expected when creating testing context")

	tCtx.CreateComputeTemplate(t)
	t.Cleanup(func() {
		tCtx.DeleteComputeTemplate(t)
	})

	response, actualRpcStatus, err := tCtx.GetRayApiServerClient().GetAllComputeTemplates()
	require.NoError(t, err, "No error expected")
	require.Nil(t, actualRpcStatus, "No RPC status expected")
	require.NotNil(t, response, "A response is expected")
	require.NotEmpty(t, response.ComputeTemplates, "A list of compute templates is required")
	foundName := false
	for _, template := range response.ComputeTemplates {
		if tCtx.GetComputeTemplateName() == template.Name && tCtx.GetNamespaceName() == template.Namespace {
			foundName = true
			break
		}
	}
	require.Equal(t, foundName, true)
}

// TestGetTemplatesInNamespace get all compute templates in namespace endpoint
// to validate various input scenarios
func TestGetTemplatesInNamespace(t *testing.T) {
	tCtx, err := NewEnd2EndTestingContext(t)
	require.NoError(t, err, "No error expected when creating testing context")

	tCtx.CreateComputeTemplate(t)
	t.Cleanup(func() {
		tCtx.DeleteComputeTemplate(t)
	})

	response, actualRpcStatus, err := tCtx.GetRayApiServerClient().GetAllComputeTemplatesInNamespace(
		&api.ListComputeTemplatesRequest{
			Namespace: tCtx.GetNamespaceName(),
		})
	require.NoError(t, err, "No error expected")
	require.Nil(t, actualRpcStatus, "No RPC status expected")
	require.NotNil(t, response, "A response is expected")
	require.NotEmpty(t, response.ComputeTemplates, "A list of compute templates is required")
	foundName := false
	for _, template := range response.ComputeTemplates {
		if tCtx.GetComputeTemplateName() == template.Name && tCtx.GetNamespaceName() == template.Namespace {
			foundName = true
			break
		}
	}
	require.Equal(t, foundName, true)
}

// TestDeleteTemplate sequentially iterates over the delete compute template endpoint
// to validate various input scenarios
func TestGetTemplateByNameInNamespace(t *testing.T) {
	tCtx, err := NewEnd2EndTestingContext(t)
	require.NoError(t, err, "No error expected when creating testing context")

	tCtx.CreateComputeTemplate(t)

	tests := []GenericEnd2EndTest[*api.GetComputeTemplateRequest]{
		{
			Name: "Get template by Name in a namespace",
			Input: &api.GetComputeTemplateRequest{
				Name:      tCtx.GetComputeTemplateName(),
				Namespace: tCtx.GetNamespaceName(),
			},
			ExpectedError: nil,
		},
		{
			Name: "Get non existing template",
			Input: &api.GetComputeTemplateRequest{
				Name:      "another-template",
				Namespace: tCtx.GetNamespaceName(),
			},
			ExpectedError: &kuberayHTTP.KuberayAPIServerClientError{
				HTTPStatusCode: http.StatusNotFound,
			},
		},
		{
			Name: "Get a template with no Name",
			Input: &api.GetComputeTemplateRequest{
				Namespace: tCtx.GetNamespaceName(),
			},
			ExpectedError: &kuberayHTTP.KuberayAPIServerClientError{
				HTTPStatusCode: http.StatusBadRequest,
			},
		},
		{
			Name: "Get a template with no namespace",
			Input: &api.GetComputeTemplateRequest{
				Name: "some-Name",
			},
			ExpectedError: &kuberayHTTP.KuberayAPIServerClientError{
				HTTPStatusCode: http.StatusNotFound,
			},
		},
	}
	// Execute tests sequentially
	for _, tc := range tests {
		tc := tc // capture range variable
		t.Run(tc.Name, func(t *testing.T) {
			actualTemplate, actualRpcStatus, err := tCtx.GetRayApiServerClient().GetComputeTemplate(tc.Input)
			if tc.ExpectedError != nil {
				require.EqualError(t, err, tc.ExpectedError.Error(), "Matching error expected")
				require.NotNil(t, actualRpcStatus, "A not nill RPC status is required")
			} else {
				require.NoError(t, err, "No error expected")
				require.Nil(t, actualRpcStatus, "No RPC status expected")
				require.Equal(t, tCtx.GetComputeTemplateName(), actualTemplate.Name)
				require.Equal(t, tCtx.GetNamespaceName(), actualTemplate.Namespace)
			}
		})
	}
}
