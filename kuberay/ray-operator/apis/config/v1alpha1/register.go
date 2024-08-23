package v1alpha1

import (
	"k8s.io/apimachinery/pkg/runtime/schema"
)

func init() {
	SchemeBuilder.Register(&Configuration{})
	SchemeBuilder.SchemeBuilder.Register(addDefaultingFuncs)
}

// SchemeGroupVersion is group version used to register these objects.
var SchemeGroupVersion = GroupVersion

func Resource(resource string) schema.GroupResource {
	return SchemeGroupVersion.WithResource(resource).GroupResource()
}
