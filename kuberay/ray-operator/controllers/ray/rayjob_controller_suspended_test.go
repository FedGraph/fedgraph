/*

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package ray

import (
	"context"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	batchv1 "k8s.io/api/batch/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/client-go/util/retry"

	rayv1 "github.com/ray-project/kuberay/ray-operator/apis/ray/v1"
	"github.com/ray-project/kuberay/ray-operator/controllers/ray/common"
)

var _ = Context("RayJob with suspend operation", func() {
	Describe("When creating a rayjob with suspend == true", func() {
		ctx := context.Background()
		namespace := "default"
		rayCluster := &rayv1.RayCluster{}
		rayJob := rayJobTemplate("rayjob-suspend", namespace)
		rayJob.Spec.Suspend = true

		It("should create a rayjob object", func() {
			err := k8sClient.Create(ctx, rayJob)
			Expect(err).NotTo(HaveOccurred(), "failed to create test RayJob resource")
		})

		It("should have deployment status suspended", func() {
			Eventually(
				getRayJobDeploymentStatus(ctx, rayJob),
				time.Second*5, time.Millisecond*500).Should(Equal(rayv1.JobDeploymentStatusSuspended))
		})

		It("should NOT create a raycluster object", func() {
			Consistently(
				getRayClusterNameForRayJob(ctx, rayJob),
				time.Second*3, time.Millisecond*500).Should(BeEmpty())
		})

		It("should unsuspend a rayjob object", func() {
			err := updateRayJobSuspendField(ctx, rayJob, false)
			Expect(err).NotTo(HaveOccurred(), "failed to update RayJob")
		})

		It("should create a raycluster object", func() {
			// Ray Cluster name can be present on RayJob's CRD
			Eventually(
				getRayClusterNameForRayJob(ctx, rayJob),
				time.Second*15, time.Millisecond*500).Should(Not(BeEmpty()))
			// The actual cluster instance and underlying resources SHOULD be created when suspend == false
			Eventually(
				// k8sClient client does not throw error if cluster IS found
				getResourceFunc(ctx, common.RayJobRayClusterNamespacedName(rayJob), rayCluster),
				time.Second*3, time.Millisecond*500).Should(BeNil())
		})

		It("should NOT create the underlying K8s job yet because the cluster is not ready", func() {
			underlyingK8sJob := &batchv1.Job{}
			Consistently(
				// k8sClient client throws error if resource not found
				func() bool {
					err := getResourceFunc(ctx, common.RayJobK8sJobNamespacedName(rayJob), underlyingK8sJob)()
					return errors.IsNotFound(err)
				},
				time.Second*3, time.Millisecond*500).Should(BeTrue())
		})

		It("should be able to update all Pods to Running", func() {
			updateHeadPodToRunningAndReady(ctx, rayJob.Status.RayClusterName, namespace)
			updateWorkerPodsToRunningAndReady(ctx, rayJob.Status.RayClusterName, namespace)
		})

		It("Dashboard URL should be set", func() {
			Eventually(
				getDashboardURLForRayJob(ctx, rayJob),
				time.Second*3, time.Millisecond*500).Should(HavePrefix(rayJob.Name), "Dashboard URL = %v", rayJob.Status.DashboardURL)
		})

		It("should create the underlying Kubernetes Job object", func() {
			underlyingK8sJob := &batchv1.Job{}
			// The underlying Kubernetes Job should be created when the RayJob is created
			Eventually(
				getResourceFunc(ctx, common.RayJobK8sJobNamespacedName(rayJob), underlyingK8sJob),
				time.Second*3, time.Millisecond*500).Should(BeNil(), "Expected Kubernetes job to be present")
		})
	})

	Describe("RayJob suspend operation shoud be atomic", func() {
		ctx := context.Background()
		namespace := "default"
		rayJob := rayJobTemplate("rayjob-atomic-suspend", namespace)
		rayCluster := &rayv1.RayCluster{}

		It("Create a RayJob custom resource", func() {
			err := k8sClient.Create(ctx, rayJob)
			Expect(err).NotTo(HaveOccurred(), "Failed to create RayJob")
		})

		It("RayJobs's JobDeploymentStatus transitions from New to Initializing.", func() {
			Eventually(
				getRayJobDeploymentStatus(ctx, rayJob),
				time.Second*3, time.Millisecond*500).Should(Equal(rayv1.JobDeploymentStatusInitializing), "JobDeploymentStatus = %v", rayJob.Status.JobDeploymentStatus)
		})

		It("Make RayCluster.Status.State to be rayv1.Ready", func() {
			updateHeadPodToRunningAndReady(ctx, rayJob.Status.RayClusterName, namespace)
			updateWorkerPodsToRunningAndReady(ctx, rayJob.Status.RayClusterName, namespace)
			Eventually(
				getClusterState(ctx, namespace, rayJob.Status.RayClusterName),
				time.Second*3, time.Millisecond*500).Should(Equal(rayv1.Ready))
		})

		It("RayJobs's JobDeploymentStatus transitions from Initializing to Running.", func() {
			Eventually(
				getRayJobDeploymentStatus(ctx, rayJob),
				time.Second*3, time.Millisecond*500).Should(Equal(rayv1.JobDeploymentStatusRunning), "JobDeploymentStatus = %v", rayJob.Status.JobDeploymentStatus)
		})

		// The finalizer here is used to prevent the RayCluster from being deleted,
		// ensuring that the RayJob remains in Suspending status once the suspend field is set to true.
		It("Add finalizer to the RayCluster", func() {
			err := retry.RetryOnConflict(retry.DefaultRetry, func() error {
				err := k8sClient.Get(ctx, common.RayJobRayClusterNamespacedName(rayJob), rayCluster)
				if err != nil {
					return err
				}
				rayCluster.Finalizers = append(rayCluster.Finalizers, "ray.io/deletion-blocker")
				return k8sClient.Update(ctx, rayCluster)
			})
			Expect(err).NotTo(HaveOccurred(), "failed to add finalizer to RayCluster")
		})

		It("Suspend the RayJob", func() {
			err := updateRayJobSuspendField(ctx, rayJob, true)
			Expect(err).NotTo(HaveOccurred(), "failed to update RayJob")
		})

		It("RayJobs's JobDeploymentStatus transitions from Running to Suspending.", func() {
			Eventually(
				getRayJobDeploymentStatus(ctx, rayJob),
				time.Second*3, time.Millisecond*500).Should(Equal(rayv1.JobDeploymentStatusSuspending), "JobDeploymentStatus = %v", rayJob.Status.JobDeploymentStatus)
		})

		// The suspend operation is atomic; regardless of how the user sets the suspend field at this moment, the status should be Suspending.
		It("Change the suspend field of RayJob from true to false and then back to true.", func() {
			err := updateRayJobSuspendField(ctx, rayJob, false)
			Expect(err).NotTo(HaveOccurred(), "failed to update RayJob")
			Consistently(
				getRayJobDeploymentStatus(ctx, rayJob),
				time.Second*3, time.Millisecond*500).Should(Equal(rayv1.JobDeploymentStatusSuspending), "JobDeploymentStatus = %v", rayJob.Status.JobDeploymentStatus)

			err = updateRayJobSuspendField(ctx, rayJob, true)
			Expect(err).NotTo(HaveOccurred(), "failed to update RayJob")
			Consistently(
				getRayJobDeploymentStatus(ctx, rayJob),
				time.Second*3, time.Millisecond*500).Should(Equal(rayv1.JobDeploymentStatusSuspending), "JobDeploymentStatus = %v", rayJob.Status.JobDeploymentStatus)
		})

		It("Remove finalizer from the RayCluster", func() {
			err := retry.RetryOnConflict(retry.DefaultRetry, func() error {
				err := k8sClient.Get(ctx, common.RayJobRayClusterNamespacedName(rayJob), rayCluster)
				if err != nil {
					return err
				}
				rayCluster.Finalizers = []string{}
				return k8sClient.Update(ctx, rayCluster)
			})
			Expect(err).NotTo(HaveOccurred(), "failed to remove finalizer from RayCluster")
		})

		It("RayJobs's JobDeploymentStatus transitions from Suspending to Suspended.", func() {
			Eventually(
				getRayJobDeploymentStatus(ctx, rayJob),
				time.Second*3, time.Millisecond*500).Should(Equal(rayv1.JobDeploymentStatusSuspended), "JobDeploymentStatus = %v", rayJob.Status.JobDeploymentStatus)
		})
	})
})
